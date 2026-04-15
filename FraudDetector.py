import json
import os
import re
from typing import Dict, List

import joblib
import numpy as np
import torch
from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer

from text_utils import build_model_text, build_turn_windows, extract_structured_features, split_transcript_lines, unique_preserve_order


DEFAULT_MULTIPLE_CHOICE_PROMPTS = [
    "Этот текст или разговор является обычным безопасным сообщением без признаков мошенничества.",
    "Этот текст или разговор является мошенничеством, скамом или социальной инженерией.",
]

BENIGN_PATTERNS = [
    re.compile(r"\bкурьер\b.*\b(час|минут|подъед|привез|достав)", re.IGNORECASE),
    re.compile(r"\b(мой|это)\b.*\bновый номер\b", re.IGNORECASE),
    re.compile(r"\b(мастер|врач|школа|салон|такси)\b.*\b(буду|приед|напомина|завтра)\b", re.IGNORECASE),
    re.compile(r"\bзаказ\b.*\b(будет|привез|достав)\b", re.IGNORECASE),
]


class FraudDetector:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        metadata_path = os.path.join(model_dir, "metadata.json")
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        self.architecture = self.metadata.get("architecture", "sequence_classification")
        self.max_length = int(self.metadata.get("max_length", 256))
        self.classes = list(self.metadata.get("classes") or joblib.load(os.path.join(model_dir, "label_encoder.joblib")))
        self.fraud_index = self.classes.index("fraud")
        self.multiple_choice_prompts = self.metadata.get("multiple_choice_prompts") or DEFAULT_MULTIPLE_CHOICE_PROMPTS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.architecture == "multiple_choice":
            self.model = AutoModelForMultipleChoice.from_pretrained(model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        prepared = [build_model_text(text, extract_structured_features(text)) for text in texts]
        if self.architecture == "multiple_choice":
            first_sentences = [[text] * len(self.multiple_choice_prompts) for text in prepared]
            second_sentences = [self.multiple_choice_prompts for _ in prepared]
            flat_first = [item for group in first_sentences for item in group]
            flat_second = [item for group in second_sentences for item in group]
            encoded = self.tokenizer(
                flat_first,
                flat_second,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            num_choices = len(self.multiple_choice_prompts)
            return {key: value.view(len(prepared), num_choices, -1).to(self.device) for key, value in encoded.items()}

        return self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            encoded = self._prepare_inputs(texts)
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        probs = self._predict_proba([text])[0]
        features = extract_structured_features(text)
        adjusted_fraud_probability, decision_reasons = self._adjust_fraud_probability(text, float(probs[self.fraud_index]), features)
        adjusted_probs = self._build_binary_probs(adjusted_fraud_probability)
        predicted_index = int(np.argmax(adjusted_probs))
        predicted_class = self.classes[predicted_index]
        confidence = float(adjusted_probs[predicted_index])
        decision = self._decision_label(adjusted_fraud_probability)
        result = {
            "predicted_class": decision,
            "model_predicted_class": predicted_class,
            "confidence": confidence,
            "is_fraud": decision == "fraud",
            "risk_level": self._risk_level(adjusted_fraud_probability),
            "decision_reasons": decision_reasons,
        }
        if return_probabilities:
            result["probabilities"] = {label: float(adjusted_probs[index]) for index, label in enumerate(self.classes)}
            result["fraud_probability"] = adjusted_fraud_probability
        return result

    def predict_batch(self, texts: List[str], return_probabilities: bool = True) -> List[Dict]:
        probs = self._predict_proba(texts)
        results = []
        for text, row in zip(texts, probs):
            features = extract_structured_features(text)
            adjusted_fraud_probability, decision_reasons = self._adjust_fraud_probability(text, float(row[self.fraud_index]), features)
            adjusted_probs = self._build_binary_probs(adjusted_fraud_probability)
            predicted_index = int(np.argmax(adjusted_probs))
            predicted_class = self.classes[predicted_index]
            confidence = float(adjusted_probs[predicted_index])
            decision = self._decision_label(adjusted_fraud_probability)
            item = {
                "predicted_class": decision,
                "model_predicted_class": predicted_class,
                "confidence": confidence,
                "is_fraud": decision == "fraud",
                "risk_level": self._risk_level(adjusted_fraud_probability),
                "decision_reasons": decision_reasons,
            }
            if return_probabilities:
                item["probabilities"] = {label: float(adjusted_probs[index]) for index, label in enumerate(self.classes)}
                item["fraud_probability"] = adjusted_fraud_probability
            results.append(item)
        return results

    def analyze_text(self, text: str) -> Dict:
        features = extract_structured_features(text)
        prediction = self.predict(text, return_probabilities=True)
        markers = self._extract_markers(text)
        return {
            "original_text": text,
            "processed_text": build_model_text(text, features),
            "features": features,
            "prediction": prediction,
            "markers": markers,
            "recommendation": self._recommendation(prediction["fraud_probability"], markers),
        }

    def analyze_call(self, transcript: str, window_size: int = 3, step: int = 1) -> Dict:
        lines = split_transcript_lines(transcript)
        if not lines:
            return {"error": "Transcript is empty", "prediction": None, "segments": []}

        windows = build_turn_windows(lines, window_size=window_size, step=step)
        segment_texts = unique_preserve_order([transcript] + lines + windows)
        segment_predictions = self.predict_batch(segment_texts, return_probabilities=True)

        segments = []
        for text, prediction in zip(segment_texts, segment_predictions):
            segments.append(
                {
                    "text": text,
                    "fraud_probability": prediction["fraud_probability"],
                    "predicted_class": prediction["predicted_class"],
                    "risk_level": prediction["risk_level"],
                    "decision_reasons": prediction.get("decision_reasons", []),
                }
            )

        suspicious_segments = sorted(
            [segment for segment in segments if segment["fraud_probability"] >= 0.55],
            key=lambda item: item["fraud_probability"],
            reverse=True,
        )[:5]

        whole_call_prob = segment_predictions[0]["fraud_probability"]
        top_segment_prob = max(segment["fraud_probability"] for segment in segments)
        mean_segment_prob = float(np.mean([segment["fraud_probability"] for segment in segments]))

        features = extract_structured_features(transcript)
        markers = self._extract_markers(transcript)
        heuristic_bonus = min(0.2, 0.03 * len(markers))
        final_probability = min(1.0, 0.5 * whole_call_prob + 0.35 * top_segment_prob + 0.15 * mean_segment_prob + heuristic_bonus)

        return {
            "transcript": transcript,
            "line_count": len(lines),
            "segment_count": len(segment_texts),
            "features": features,
            "fraud_probability": final_probability,
            "predicted_class": self._decision_label(final_probability),
            "risk_level": self._risk_level(final_probability),
            "markers": markers,
            "suspicious_segments": suspicious_segments,
            "whole_call_prediction": segment_predictions[0],
            "decision_reasons": segment_predictions[0].get("decision_reasons", []),
            "recommendation": self._recommendation(final_probability, markers),
        }

    def get_model_info(self) -> Dict:
        return {
            "model_dir": self.model_dir,
            "classes": self.classes,
            "metadata": self.metadata,
        }

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.9:
            return "CRITICAL"
        if probability >= 0.75:
            return "HIGH"
        if probability >= 0.55:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _decision_label(probability: float) -> str:
        if probability >= 0.75:
            return "fraud"
        if probability >= 0.45:
            return "suspicious"
        return "normal"

    @staticmethod
    def _build_binary_probs(fraud_probability: float) -> np.ndarray:
        return np.array([1.0 - fraud_probability, fraud_probability], dtype=np.float32)

    def _adjust_fraud_probability(self, text: str, fraud_probability: float, features: Dict):
        adjusted = fraud_probability
        text_lower = text.lower()
        reasons = [f"Базовая вероятность модели: {fraud_probability:.3f}"]
        high_risk_signals = (
            int(features["has_code_request"])
            + int(features["has_money_transfer_request"])
            + int(features["has_sensitive_data_request"])
            + int(features["has_remote_access_request"])
        )

        if high_risk_signals == 0:
            adjusted -= 0.18
            reasons.append("Снижение риска: нет запроса кода, денег, чувствительных данных или удалённого доступа.")

        if int(features["has_authority_impersonation"]) == 0 and int(features["has_threat"]) == 0:
            adjusted -= 0.08
            reasons.append("Снижение риска: нет имитации банка/службы и нет угрозы блокировки или списания.")

        if len(text_lower.split()) <= 14 and int(features["risk_markers_count"]) <= 1:
            adjusted -= 0.1
            reasons.append("Снижение риска: текст короткий и почти без опасных маркеров.")

        if any(pattern.search(text_lower) for pattern in BENIGN_PATTERNS):
            adjusted -= 0.2
            reasons.append("Снижение риска: найден бытовой безопасный шаблон вроде курьера, нового номера или сервисного звонка.")

        if features["scenario_type"] == "delivery_fee" and high_risk_signals == 0:
            adjusted -= 0.08
            reasons.append("Снижение риска: доставка без требования кода или перевода денег.")

        if features["scenario_type"] == "benign_service" and int(features["risk_markers_count"]) == 0:
            adjusted -= 0.1
            reasons.append("Снижение риска: сценарий похож на обычный сервисный контакт без опасных действий.")

        if int(features["has_code_request"]):
            reasons.append("Повышение риска: есть запрос кода из SMS или подтверждения.")
        if int(features["has_money_transfer_request"]):
            reasons.append("Повышение риска: есть просьба перевести или оплатить деньги.")
        if int(features["has_sensitive_data_request"]):
            reasons.append("Повышение риска: запрашиваются чувствительные данные.")
        if int(features["has_remote_access_request"]):
            reasons.append("Повышение риска: есть просьба установить ПО удалённого доступа.")
        if int(features["has_threat"]):
            reasons.append("Повышение риска: есть угроза блокировки, списания или кредита.")
        if int(features["has_authority_impersonation"]):
            reasons.append("Повышение риска: есть имитация банка, поддержки или госоргана.")

        adjusted = max(0.01, min(0.99, adjusted))
        reasons.append(f"Итоговая вероятность после коррекции: {adjusted:.3f}")
        return adjusted, reasons

    def _extract_markers(self, text: str) -> List[str]:
        features = extract_structured_features(text)
        markers = [name for name, enabled in features.items() if name.startswith("has_") and int(enabled) == 1]
        markers.append(f"scenario:{features['scenario_type']}")
        markers.append(f"stage:{features['fraud_stage']}")
        if features["victim_confused_or_resisting"]:
            markers.append("victim_confused_or_resisting")
        return markers

    def _recommendation(self, fraud_probability: float, markers: List[str]) -> str:
        if fraud_probability >= 0.9:
            return "Критический риск. Такой звонок лучше помечать как мошеннический и отправлять на ручную проверку."
        if fraud_probability >= 0.75:
            return "Высокий риск. Рекомендую отметить звонок как подозрительный и показать оператору ключевые фрагменты."
        if fraud_probability >= 0.45 or len(markers) >= 2:
            return "Пограничный случай. Лучше отправить разговор на ручную проверку, но не считать его мошенническим автоматически."
        return "Низкий риск. Явных признаков мошенничества недостаточно, разговор можно считать нормальным."


if __name__ == "__main__":
    model_path = os.environ.get("FRAUD_MODEL_DIR", "")
    if not model_path:
        candidates = sorted(
            [item for item in os.listdir(".") if item.startswith("fraud_call_model_") and os.path.isdir(item)],
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No trained model directory found. Train the model first or set FRAUD_MODEL_DIR.")
        model_path = candidates[0]
    detector = FraudDetector(model_path)
    example = """Мошенник - Здравствуйте
Ответчик - Здравствуйте
Мошенник - Я звоню из банка техподдержки
Ответчик - Да слушаю
Мошенник - С вашего аккаунта сняли 10000 тенге. Чтобы заблокировать операцию скажите код из смс"""
    result = detector.analyze_call(example)
    print(json.dumps(result, ensure_ascii=False, indent=2))
