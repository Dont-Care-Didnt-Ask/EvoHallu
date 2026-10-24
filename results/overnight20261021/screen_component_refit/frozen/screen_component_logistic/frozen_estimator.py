class HallucinationDetector(Estimator):
    def __init__(self, context):
        from scripts.overnight_component_model import FrozenComponentLogistic
        self.frozen_head = FrozenComponentLogistic(context, 'results/overnight20261021/screen_component_refit/weights.json', '0319dd55eb20c87e20596d892fb2e384451126d80bebc7c45989a41a1d9602e3', 'e86f98e49d710317e685af1a57f5ee68ba4b1fd48ac5a919091f0cc0eb564a01')
        super().__init__(context, stats_dependencies=self.frozen_head.stats_dependencies, level="sequence")

    def __call__(self, stats):
        return self.frozen_head(stats)
