import json, pathlib
cs = [0.0, -0.0, 1.0, 100.0, 3.0, 1.5, 0.0001, 0.0001234, 1e-5, -1.5e-7,
      1e15, 1e16, -2.5e20, 1234.5, 3.141592653589793,
      0.3333333333333333, 2.2250738585072014e-308, 5e-324, 0.1,
      1e-4, 9.999999999999999e15, 1.0000000000000002, -1e-4]
cs += [0, 1, -1, 42, 9223372036854775807, -9223372036854775808, 253]
cs += ["", "plain", 'quote " backslash \\', "new\nline\ttab\rret",
       "bell\bform\f", " ctrl\x01\x1f", " del\x7f kept",
       "héllo wörld", "こんにちは", "flags 🇸🇪",
       "</script> & <tag>", "  spaces  ", "x"]
cs += [{"b": 1, "a": 2, "c": {"z": None, "y": [True, False]}},
       [], {}, [None, [None, []]], {"": "empty key"},
       {"ключ": "значение", "键": "值"},
       [[[[[[[[[1]]]]]]]]], {"n": [0.0, -0.0, 1e16, 1.0]}]
cs += [
    {"state": "My card was charged twice.", "model": "jev-latest",
     "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}},
    {"state": {"ticket": {"subject": "Duplicate charge",
                          "messages": [{"from": "customer", "text": "Charged twice for A-104."},
                                       {"from": "support", "text": "Checking."}],
                          "flags": [True, False, None]},
               "order": {"id": "A-104", "charges": [49, 49.0, -0.5]}},
     "questions": {"dept": {"type": "choice", "instructions": "Which team?",
                            "criteria": {"billing": "refunds", "technical": None, "sales": "pricing"}},
                   "score_it": {"type": "score", "criteria": ["Calm", "Angry"]}}},
    {"state": "héllo wörld — नमस्ते — こんにちは",
     "questions": {"q": {"type": "noul", "instructions": "Mixed script?\n\tYes or no"}}},
    {"state": [1, 2.5, "three", [4, [5, {"six": None}]]], "questions": {}},
]
d = pathlib.Path(__file__).resolve().parent
(d / "pyjson_cases.json").write_text(json.dumps(cs, ensure_ascii=False, indent=0))
(d / "pyjson_py.json").write_text(json.dumps(
    [json.dumps(c, ensure_ascii=False) for c in cs], ensure_ascii=False, indent=0))
print(len(cs), "cases -> pyjson_cases.json + pyjson_py.json")

