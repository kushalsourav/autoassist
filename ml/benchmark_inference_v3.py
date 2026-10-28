"""CPU-only inference benchmark for locally fine-tuned multi-label DistilBERT."""
import argparse
import statistics
import time


def percentile(sorted_values, p):
    index = (len(sorted_values) - 1) * p / 100
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] * (high - index) + sorted_values[high] * (index - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error('--runs must be at least 2')
    from backend.app.classifier import _predict_multilabel_v3, _load_multilabel_v3
    if _load_multilabel_v3() is None:
        raise SystemExit('Missing V3 model weights/config/metrics at ml/artifacts/multilabel_distilbert_v3')
    complaint = ('When accelerating uphill, the engine hesitates, the check-engine '
                 'light turns on, and the transmission sometimes jerks.')
    for _ in range(3):
        _predict_multilabel_v3(complaint)
    measured = []
    for _ in range(args.runs):
        start = time.perf_counter()
        result = _predict_multilabel_v3(complaint)
        measured.append((time.perf_counter() - start) * 1000)
    times = sorted(measured)
    print('Model:', result['method'])
    print('Selected labels:', result['labels'])
    print('CPU latency (ms): mean=%.1f median=%.1f p95=%.1f' %
          (statistics.mean(times), statistics.median(times), percentile(times, 95)))
    print('Cold model-load time is excluded. These are local single-request measurements.')


if __name__ == '__main__':
    main()
