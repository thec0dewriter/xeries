import pathlib

def run():
    p1 = pathlib.Path('tests/unit/test_permutation.py')
    content1 = p1.read_text(encoding='utf-8')
    content1 = content1.replace('.compute(', '.explain(')
    p1.write_text(content1, encoding='utf-8')

    p2 = pathlib.Path('tests/integration/test_skforecast.py')
    content2 = p2.read_text(encoding='utf-8')
    content2 = content2.replace('.compute(', '.explain(')
    p2.write_text(content2, encoding='utf-8')

    p3 = pathlib.Path('tests/unit/test_shap.py')
    content3 = p3.read_text(encoding='utf-8')
    content3 = content3.replace('predict_fn=fitted_model_small.predict', 'model=fitted_model_small')
    p3.write_text(content3, encoding='utf-8')

run()
