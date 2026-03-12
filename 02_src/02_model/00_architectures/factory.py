from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class LRBuilder:
    def build(self, params):
        model = LogisticRegression(**params)
        return model

class RFBuilder:
    def build(self, params):
        model = RandomForestClassifier(**params)
        return model

class XGBBuilder:
    def build(self, params):
        model = XGBClassifier(**params)
        return model


class ModelFactory:
    _builders = {
        "RandomForest": RFBuilder(),
        "XGBoost": XGBBuilder(),
        "LogisticRegression": LRBuilder()
    }

    @classmethod
    def create_model(cls, model_name, params):
        builder = cls._builders.get(model_name)
        if not builder:
            raise ValueError(f"❌ 지원하지 않는 모델입니다: {model_name}")
        return builder.build(params)