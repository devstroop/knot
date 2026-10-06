//! Candle native runtime (M8). Runs the same decision model the ONNX path
//! serves — ModernBERT encoder plus the typed decision head — without an
//! ONNX Runtime dependency.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert;

use crate::error::{Error, Result};
use crate::runtime::{Calibration, Runtime};

fn err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

#[derive(Default)]
struct CandleProfile {
    calls: u64,
    tensor_prep: Duration,
    encoder: Duration,
    decision_head: Duration,
    classifier_and_copy: Duration,
}

impl CandleProfile {
    fn report(&self) {
        if self.calls == 0 {
            return;
        }
        let avg_ms = |duration: Duration| duration.as_secs_f64() * 1e3 / self.calls as f64;
        eprintln!(
            "candle_profile calls={} avg_ms tensor_prep={:.2} encoder={:.2} decision_head={:.2} classifier_and_copy={:.2}",
            self.calls,
            avg_ms(self.tensor_prep),
            avg_ms(self.encoder),
            avg_ms(self.decision_head),
            avg_ms(self.classifier_and_copy),
        );
    }
}

/// Mirror Laya's `_apply_rope_config`: transformers 5 stores per-layer-type
/// RoPE thetas under `rope_parameters`; candle wants flat fields.
fn patch_rope(cfg: &mut serde_json::Value) {
    let rope = match cfg.get("rope_parameters") {
        Some(r) => r.clone(),
        None => return,
    };
    let flat = rope.get("rope_theta").and_then(|v| v.as_f64());
    let mut set = |layer_type: &str, attr: &str| {
        let theta = rope
            .get(layer_type)
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or(flat);
        if let Some(t) = theta {
            cfg[attr] = serde_json::json!(t);
        }
    };
    set("full_attention", "global_rope_theta");
    set("sliding_attention", "local_rope_theta");
}

/// Standard `nn.TransformerEncoderLayer` (norm_first, batch_first, eval).
struct HeadLayer {
    in_proj: Linear,
    out_proj: Linear,
    nhead: usize,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl HeadLayer {
    fn load(vb: VarBuilder, d: usize, nhead: usize) -> Result<Self> {
        let captured_nhead = nhead;
        let in_proj_w = vb
            .get((3 * d, d), "self_attn.in_proj_weight")
            .map_err(|e| err(format!("in_proj_weight: {e}")))?;
        let in_bias = vb
            .get((3 * d,), "self_attn.in_proj_bias")
            .map_err(|e| err(format!("in_proj_bias: {e}")))?;
        let out_w = vb
            .get((d, d), "self_attn.out_proj.weight")
            .map_err(|e| err(format!("out_proj.weight: {e}")))?;
        let out_b = vb
            .get((d,), "self_attn.out_proj.bias")
            .map_err(|e| err(format!("out_proj.bias: {e}")))?;
        let linear1 = candle_nn::linear(d, 4 * d, vb.pp("linear1"))
            .map_err(|e| err(format!("linear1: {e}")))?;
        let linear2 = candle_nn::linear(4 * d, d, vb.pp("linear2"))
            .map_err(|e| err(format!("linear2: {e}")))?;
        let norm1 = candle_nn::layer_norm(d, 1e-5, vb.pp("norm1"))
            .map_err(|e| err(format!("norm1: {e}")))?;
        let norm2 = candle_nn::layer_norm(d, 1e-5, vb.pp("norm2"))
            .map_err(|e| err(format!("norm2: {e}")))?;
        Ok(Self {
            in_proj: Linear::new(in_proj_w, Some(in_bias)),
            nhead: captured_nhead,
            out_proj: Linear::new(out_w, Some(out_b)),
            linear1,
            linear2,
            norm1,
            norm2,
        })
    }

    fn forward(&self, xs: &Tensor, pad: &Tensor) -> Result<Tensor> {
        let mha_out = self.mha(&xs.apply(&self.norm1)?, pad)?;
        let xs = (xs + mha_out)?;
        let mlp = self
            .linear2
            .forward(&self.linear1.forward(&xs.apply(&self.norm2)?)?.gelu_erf()?)?;
        Ok((xs + mlp)?)
    }

    fn mha(&self, xs: &Tensor, pad: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let nhead = self.nhead;
        let qkv = self.in_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;
        let q = q.reshape((b, t, nhead, d / nhead))?.transpose(1, 2)?;
        let k = k.reshape((b, t, nhead, d / nhead))?.transpose(1, 2)?;
        let v = v.reshape((b, t, nhead, d / nhead))?.transpose(1, 2)?;
        let scale = (d as f64 / nhead as f64).powf(-0.5);
        let att = (q * scale)?.matmul(&k.t()?.contiguous()?)?;
        let att = att.broadcast_add(pad)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

pub struct CandleRuntime {
    device: Device,
    encoder: modernbert::ModernBert,
    type_emb: Embedding,
    head_layers: Vec<HeadLayer>,
    scorer: (LayerNorm, Linear, Linear),
    act_head: (Linear, Linear),
    config: serde_json::Value,
    calibration: Calibration,
    profile: Option<Mutex<CandleProfile>>,
}

impl CandleRuntime {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let profile = match std::env::var("KNOT_CANDLE_PROFILE") {
            Ok(value) if value == "1" || value == "true" => {
                Some(Mutex::new(CandleProfile::default()))
            }
            Ok(value) if value == "0" || value == "false" => None,
            Ok(value) => {
                return Err(err(format!(
                    "KNOT_CANDLE_PROFILE must be 0/1 or false/true, got {value:?}"
                )));
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(err(format!("read KNOT_CANDLE_PROFILE: {e}"))),
        };
        let device = Device::Cpu;
        let cfg_text =
            std::fs::read_to_string(model_dir.join("rl_agent_config.json")).map_err(Error::Io)?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;
        let calibration = Calibration::from_config(&cfg);
        let head_layers_n = cfg.get("head_layers").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

        let mut enc_cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("encoder/config.json")).map_err(Error::Io)?,
        )?;
        patch_rope(&mut enc_cfg);
        let enc_config: modernbert::Config = serde_json::from_value(enc_cfg.clone())
            .map_err(|e| err(format!("parse ModernBERT config: {e}")))?;

        let mut tensors =
            candle_core::safetensors::load(model_dir.join("model.safetensors"), &device)
                .map_err(|e| err(format!("load safetensors: {e}")))?;
        // Candle expects "model.*" prefixes for the backbone.
        let mut remapped = HashMap::new();
        for (k, v) in tensors.drain() {
            let key = if let Some(rest) = k.strip_prefix("encoder.") {
                format!("model.{rest}")
            } else {
                k
            };
            remapped.insert(key, v);
        }
        let vb = VarBuilder::from_tensors(remapped, DType::F32, &device);
        let encoder = modernbert::ModernBert::load(vb.clone(), &enc_config)
            .map_err(|e| err(format!("ModernBert::load: {e}")))?;

        let d = enc_config.hidden_size;
        let type_emb = Embedding::new(
            vb.get((3, d), "type_emb.weight")
                .map_err(|e| err(format!("type_emb: {e}")))?,
            d,
        );
        let mut head_layers = Vec::new();
        for i in 0..head_layers_n {
            head_layers.push(
                HeadLayer::load(vb.pp(format!("head.layers.{i}")), d, d / 64)
                    .map_err(|e| err(format!("head layer {i}: {e}")))?,
            );
        }
        let scorer_ln = candle_nn::layer_norm(d, 1e-5, vb.pp("scorer.0"))
            .map_err(|e| err(format!("scorer.0: {e}")))?;
        let scorer_l1 = candle_nn::linear(d, d, vb.pp("scorer.1"))
            .map_err(|e| err(format!("scorer.1: {e}")))?;
        let scorer_l2 = candle_nn::linear(d, 1, vb.pp("scorer.3"))
            .map_err(|e| err(format!("scorer.3: {e}")))?;
        let act0 = candle_nn::linear(d + 4, 256, vb.pp("act_head.0"))
            .map_err(|e| err(format!("act_head.0: {e}")))?;
        let act2 = candle_nn::linear(256, cfg_act_count(&cfg), vb.pp("act_head.2"))
            .map_err(|e| err(format!("act_head.2: {e}")))?;

        Ok(Self {
            device,
            encoder,
            type_emb,
            head_layers,
            scorer: (scorer_ln, scorer_l1, scorer_l2),
            act_head: (act0, act2),
            config: cfg,
            calibration,
            profile,
        })
    }

    fn record_profile(
        &self,
        tensor_prep: Duration,
        encoder: Duration,
        decision_head: Duration,
        classifier_and_copy: Duration,
    ) {
        if let Some(profile) = &self.profile {
            let mut profile = profile.lock().unwrap();
            profile.calls += 1;
            profile.tensor_prep += tensor_prep;
            profile.encoder += encoder;
            profile.decision_head += decision_head;
            profile.classifier_and_copy += classifier_and_copy;
        }
    }
}

impl Drop for CandleRuntime {
    fn drop(&mut self) {
        if let Some(profile) = &self.profile {
            match profile.lock() {
                Ok(profile) => profile.report(),
                Err(e) => eprintln!("candle_profile: failed to read profile: {e}"),
            }
        }
    }
}

fn cfg_act_count(cfg: &serde_json::Value) -> usize {
    cfg.get("act_costs")
        .and_then(|v| v.as_object())
        .map(|m| m.len() + 1)
        .unwrap_or(2)
}

impl Runtime for CandleRuntime {
    fn forward(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        marker_pos: &[i64],
        marker_mask: &[bool],
        qtype: &[i64],
        batch: usize,
        seq_len: usize,
        num_markers: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let device = &self.device;
        let tensor_prep_started = self.profile.as_ref().map(|_| Instant::now());
        let input_ids = Tensor::from_slice(input_ids, (batch, seq_len), device)
            .map_err(|e| err(format!("input_ids: {e}")))?
            .to_dtype(DType::U32)
            .map_err(|e| err(format!("cast ids: {e}")))?;
        let att = Tensor::from_slice(attention_mask, (batch, seq_len), device)
            .map_err(|e| err(format!("attention_mask: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| err(format!("cast att: {e}")))?;
        let marker_pos_t = Tensor::from_slice(marker_pos, (batch, num_markers), device)
            .map_err(|e| err(format!("marker_pos: {e}")))?
            .to_dtype(DType::I64)
            .map_err(|e| err(format!("cast mpos: {e}")))?;
        let qtype_t = Tensor::from_slice(qtype, (batch,), device)
            .map_err(|e| err(format!("qtype: {e}")))?
            .to_dtype(DType::U32)
            .map_err(|e| err(format!("cast qtype: {e}")))?;

        let tensor_prep = tensor_prep_started.map_or(Duration::ZERO, |t| t.elapsed());
        let encoder_started = self.profile.as_ref().map(|_| Instant::now());
        let h = self
            .encoder
            .forward(&input_ids, &att)
            .map_err(|e| err(format!("encoder: {e}")))?;
        let encoder = encoder_started.map_or(Duration::ZERO, |t| t.elapsed());
        let decision_head_started = self.profile.as_ref().map(|_| Instant::now());
        let te = self.type_emb.forward(&qtype_t)?.unsqueeze(1)?;
        let h = h
            .broadcast_add(&te)
            .map_err(|e| err(format!("type_emb add: {e}")))?;

        let mut cur = h;
        for layer in &self.head_layers {
            // key_padding_mask: additive 0 / -1e4 over padded keys
            let pad_additive = (1.0f64 - att.clone().to_dtype(DType::F32)?)?
                .reshape((batch, 1, 1, seq_len))?
                .affine(-1e4, 0.0)?;
            cur = layer
                .forward(&cur, &pad_additive)
                .map_err(|e| err(format!("head: {e}")))?;
        }

        let decision_head = decision_head_started.map_or(Duration::ZERO, |t| t.elapsed());
        let classifier_started = self.profile.as_ref().map(|_| Instant::now());
        let d = cur.dim(2)?;
        let idx = marker_pos_t
            .unsqueeze(2)?
            .expand((batch, num_markers, d))
            .map_err(|e| err(format!("expand: {e}")))?
            .contiguous()?;
        let m = cur
            .gather(&idx, 1)
            .map_err(|e| err(format!("gather: {e}")))?;
        let logits = self
            .scorer
            .2
            .forward(
                &self
                    .scorer
                    .1
                    .forward(&self.scorer.0.forward(&m)?)?
                    .gelu_erf()?,
            )?
            .squeeze(2)?;
        let mask_t = Tensor::from_slice(
            &marker_mask
                .iter()
                .map(|b| if *b { 1.0f32 } else { 0.0f32 })
                .collect::<Vec<f32>>(),
            (batch, num_markers),
            device,
        )
        .map_err(|e| err(format!("mask: {e}")))?;
        let logits = (logits + (1.0f64 - mask_t.clone())? * (-1e4f64))
            .map_err(|e| err(format!("mask logits: {e}")))?;

        // Action head features.
        let p = candle_nn::ops::softmax(&logits, D::Minus1)
            .map_err(|e| err(format!("softmax: {e}")))?;
        let k_count = mask_t.sum(D::Minus1)?;
        let k_clamped = k_count.maximum(&Tensor::full(2.0f32, k_count.shape(), device)?)?;
        let ent = {
            let lp = p.clamp(1e-9f64, 1.0f64)?.log()?;
            let neg = (p.clone() * &lp)?;
            let sum = neg.sum(D::Minus1)?;
            let denom = k_clamped.log()?;
            (sum * -1.0f64)? / denom
        }?;
        let (sorted, _idx) = p.sort_last_dim(false)?;
        let t1 = sorted.narrow(D::Minus1, 0, 1)?;
        let t2 = if num_markers >= 2 {
            sorted.narrow(D::Minus1, 1, 1)?
        } else {
            Tensor::zeros_like(&t1)?
        };
        let gap = (t1.clone() - t2)?;
        let k_norm = (k_clamped / 255.0f64)?;
        let feats = Tensor::stack(&[t1.squeeze(1)?, gap.squeeze(1)?, ent, k_norm], 1)
            .map_err(|e| err(format!("feats: {e}")))?;
        let pooled = cur.narrow(1, 0, 1)?.squeeze(1)?;
        let act_in = Tensor::cat(&[pooled, feats], 1).map_err(|e| err(format!("cat: {e}")))?;
        let act_logits = self
            .act_head
            .1
            .forward(&self.act_head.0.forward(&act_in)?.gelu_erf()?)?;

        let logits = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2())
            .map_err(|e| err(format!("extract logits: {e}")))?;
        let act = act_logits
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec2())
            .map_err(|e| err(format!("extract act: {e}")))?;
        let classifier_and_copy = classifier_started.map_or(Duration::ZERO, |t| t.elapsed());
        self.record_profile(tensor_prep, encoder, decision_head, classifier_and_copy);
        Ok((logits, act))
    }

    fn config(&self) -> &serde_json::Value {
        &self.config
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }
}
