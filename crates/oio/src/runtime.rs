//! ONNX inference path (M2). Loads Laya ONNX exports, runs the collated batch,
//! and applies per-type/per-bucket calibration temperatures.

use std::collections::BTreeMap;

#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use std::sync::Mutex;

use crate::error::Result;

#[cfg(feature = "onnx")]
use crate::error::Error;

/// Uniform inference backend behind `Engine`: one collated forward pass,
/// returning per-row marker logits and action logits.
pub trait Runtime: Send + Sync {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)>;
    fn config(&self) -> &serde_json::Value;
    fn calibration(&self) -> &Calibration;
}

/// Temperature clamp range, mirroring Laya (`TEMP_MIN`/`TEMP_MAX`).
pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

pub fn clamp_temperature(t: f64) -> f64 {
    if t.is_nan() {
        return 1.0;
    }
    t.clamp(TEMP_MIN, TEMP_MAX)
}

pub fn temp_bucket(qtype: &str, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{qtype}:{size}")
}

/// Per-checkpoint calibration: 3 per-type temperatures plus optional per-bucket map.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub temperature: [f64; 3],
    pub temperature_by_options: BTreeMap<String, f64>,
}

impl Default for Calibration {
    /// Neutral calibration: `Default`-derived would be all-zero temperatures,
    /// and `for_question` would divide by zero into NaN probabilities.
    fn default() -> Self {
        Self {
            temperature: [1.0; 3],
            temperature_by_options: BTreeMap::new(),
        }
    }
}

impl Calibration {
    pub fn from_config(cfg: &serde_json::Value) -> Self {
        let mut temperature = [1.0; 3];
        if let Some(arr) = cfg.get("temperature").and_then(|v| v.as_array()) {
            for (i, v) in arr.iter().take(3).enumerate() {
                if let Some(f) = v.as_f64() {
                    temperature[i] = clamp_temperature(f);
                }
            }
        }
        let mut by = BTreeMap::new();
        if let Some(obj) = cfg
            .get("temperature_by_options")
            .and_then(|v| v.as_object())
        {
            for (k, v) in obj {
                if let Some(f) = v.as_f64() {
                    by.insert(k.clone(), clamp_temperature(f));
                }
            }
        }
        Self {
            temperature,
            temperature_by_options: by,
        }
    }

    /// Effective temperature for a question: per-bucket override, else per-type.
    /// Clamped, so a hand-built `Calibration` with a zero/negative temperature
    /// cannot divide by zero (construction from config clamps as well).
    pub fn for_question(&self, qtype: &str, qtype_idx: usize, k: usize) -> f64 {
        let t = self
            .temperature_by_options
            .get(&temp_bucket(qtype, k))
            .copied()
            .unwrap_or(self.temperature[qtype_idx]);
        clamp_temperature(t)
    }
}

/// Softmax over `logits` scaled by `temperature`.
pub fn scaled_softmax(logits: &[f32], temperature: f64) -> Vec<f32> {
    let t = temperature as f32;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| ((z / t) - max / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// 1 − normalized entropy confidence (Laya `confidence_from_probs`).
pub fn confidence_from_probs(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let h: f32 = p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.ln()).sum();
    (1.0 - h / (k as f32).ln()).max(0.0)
}

pub fn answer_confidence(p: &[f32]) -> f32 {
    p.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(feature = "onnx")]
pub struct OnnxRuntime {
    session: Mutex<ort::session::Session>,
    pub config: serde_json::Value,
    pub calibration: Calibration,
}

#[cfg(feature = "onnx")]
impl Runtime for OnnxRuntime {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
        OnnxRuntime::forward(
            self,
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
            batch,
            seq_len,
            num_markers,
        )
    }

    fn config(&self) -> &serde_json::Value {
        &self.config
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }
}

#[cfg(feature = "onnx")]
impl OnnxRuntime {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let onnx_path = model_dir.join("laya.onnx");
        let session = ort::session::Session::builder()
            .map_err(|e| Error::Model(format!("ort builder: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Model(format!("load {}: {e}", onnx_path.display())))?;
        let cfg_path = model_dir.join("rl_agent_config.json");
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .map_err(|e| Error::Model(format!("read {}: {e}", cfg_path.display())))?,
        )?;
        let calibration = Calibration::from_config(&cfg);
        Ok(Self {
            session: Mutex::new(session),
            config: cfg,
            calibration,
        })
    }

    /// Run the model on a collated batch. Returns (logits, act_logits).
    #[allow(clippy::type_complexity)]
    pub fn forward(
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
        #![allow(clippy::too_many_arguments)]
        use ort::value::Tensor;

        let mk = |data: Vec<i64>, shape: Vec<usize>| -> Result<Tensor<i64>> {
            Tensor::from_array((shape, data.into_boxed_slice()))
                .map_err(|e| Error::Model(format!("tensor: {e}")))
        };
        let ids = mk(input_ids.to_vec(), vec![batch, seq_len])?;
        let att = mk(attention_mask.to_vec(), vec![batch, seq_len])?;
        let mpos = mk(marker_pos.to_vec(), vec![batch, num_markers])?;
        let mmask = Tensor::from_array((
            vec![batch, num_markers],
            marker_mask.to_vec().into_boxed_slice(),
        ))
        .map_err(|e| Error::Model(format!("tensor: {e}")))?;
        let qt = mk(qtype.to_vec(), vec![batch])?;

        let mut session = self.session.lock().unwrap();
        let mut out = session
            .run(ort::inputs![
                "input_ids" => &ids,
                "attention_mask" => &att,
                "marker_pos" => &mpos,
                "marker_mask" => &mmask,
                "qtype" => &qt,
            ])
            .map_err(|e| Error::Model(format!("run: {e}")))?;

        let (lshape, logits_data) = {
            let t = out
                .get_mut("logits")
                .ok_or_else(|| Error::Model("missing logits output".into()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("logits: {e}")))?;
            (
                t.0.iter().map(|&d| d as usize).collect::<Vec<usize>>(),
                t.1.to_vec(),
            )
        };
        let (ashape, act_data) = {
            let t = out
                .get_mut("act_logits")
                .ok_or_else(|| Error::Model("missing act_logits output".into()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("act_logits: {e}")))?;
            (
                t.0.iter().map(|&d| d as usize).collect::<Vec<usize>>(),
                t.1.to_vec(),
            )
        };
        let ldims: Vec<usize> = lshape;
        let logits_data = &logits_data[..];
        let logits: Vec<Vec<f32>> = if ldims.len() == 2 {
            (0..ldims[0])
                .map(|b| logits_data[b * ldims[1]..(b + 1) * ldims[1]].to_vec())
                .collect()
        } else {
            return Err(Error::Model(format!("unexpected logits shape {ldims:?}")));
        };
        let adims: Vec<usize> = ashape;
        let act_data = &act_data[..];
        let act: Vec<Vec<f32>> = if adims.len() == 2 {
            (0..adims[0])
                .map(|b| act_data[b * adims[1]..(b + 1) * adims[1]].to_vec())
                .collect()
        } else if adims.len() == 1 {
            act_data.iter().map(|&v| vec![v]).collect()
        } else {
            return Err(Error::Model(format!(
                "unexpected act_logits shape {adims:?}"
            )));
        };
        Ok((logits, act))
    }
}
