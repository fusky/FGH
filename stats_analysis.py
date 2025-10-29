import argparse
import csv
import math
from typing import List, Dict

try:
	import numpy as np
	except_available = True
except Exception:
	except_available = False
	np = None

try:
	from scipy import stats
	scipy_available = True
except Exception:
	scipy_available = False
	stats = None


def _read_csv_scores(path: str, metric: str, method: str = None, c_value: float = None) -> List[float]:
	values: List[float] = []
	with open(path, 'r', newline='') as f:
		reader = csv.DictReader(f)
		for row in reader:
			if method is not None and row.get('method') != str(method):
				continue
			if c_value is not None:
				try:
					if float(row.get('c')) != float(c_value):
						continue
				except Exception:
					continue
			val = row.get(metric)
			if val is None:
				continue
			try:
				values.append(float(val))
			except Exception:
				continue
	return values


def _mean_std(values: List[float]):
	if not values:
		return float('nan'), float('nan')
	m = sum(values) / len(values)
	var = sum((v - m) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
	return m, math.sqrt(var)


def cohens_d(sample_a: List[float], sample_b: List[float]) -> float:
	if not sample_a or not sample_b:
		return float('nan')
	mean_a, mean_b = sum(sample_a)/len(sample_a), sum(sample_b)/len(sample_b)
	var_a = sum((x - mean_a) ** 2 for x in sample_a) / (len(sample_a) - 1) if len(sample_a) > 1 else 0.0
	var_b = sum((x - mean_b) ** 2 for x in sample_b) / (len(sample_b) - 1) if len(sample_b) > 1 else 0.0
	# pooled std (unbiased)
	n_a, n_b = len(sample_a), len(sample_b)
	pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(1, (n_a + n_b - 2))
	pooled_std = math.sqrt(max(0.0, pooled_var))
	if pooled_std == 0.0:
		return 0.0
	return (mean_a - mean_b) / pooled_std


def cliffs_delta(sample_a: List[float], sample_b: List[float]) -> float:
	# nonparametric effect size
	if not sample_a or not sample_b:
		return float('nan')
	greater = 0
	less = 0
	for a in sample_a:
		for b in sample_b:
			if a > b:
				greater += 1
			elif a < b:
				less += 1
	n = len(sample_a) * len(sample_b)
	return (greater - less) / n if n > 0 else float('nan')


def main():
	parser = argparse.ArgumentParser(description='Statistical significance analysis for experimental results')
	parser.add_argument('--group_a', type=str, required=True, help='CSV path for group A results')
	parser.add_argument('--group_b', type=str, required=True, help='CSV path for group B results')
	parser.add_argument('--metric', type=str, default='best_val_acc', help='Metric column to compare')
	parser.add_argument('--method_a', type=str, default=None, help='Optional filter: method name for A')
	parser.add_argument('--method_b', type=str, default=None, help='Optional filter: method name for B')
	parser.add_argument('--c_a', type=float, default=None, help='Optional filter: curvature c for A')
	parser.add_argument('--c_b', type=float, default=None, help='Optional filter: curvature c for B')
	args = parser.parse_args()

	vals_a = _read_csv_scores(args.group_a, args.metric, method=args.method_a, c_value=args.c_a)
	vals_b = _read_csv_scores(args.group_b, args.metric, method=args.method_b, c_value=args.c_b)

	mean_a, std_a = _mean_std(vals_a)
	mean_b, std_b = _mean_std(vals_b)
	print(f'Group A (n={len(vals_a)}): mean={mean_a:.6f}, std={std_a:.6f}')
	print(f'Group B (n={len(vals_b)}): mean={mean_b:.6f}, std={std_b:.6f}')

	if scipy_available and len(vals_a) > 1 and len(vals_b) > 1:
		t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
		print(f'Welch t-test: t={t_stat:.6f}, p={p_val:.6g}')
	else:
		print('Welch t-test: unavailable (need scipy and n>1 for both groups)')

	if scipy_available and len(vals_a) > 0 and len(vals_b) > 0:
		try:
			u_stat, p_val_u = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
			print(f'Mann-Whitney U: U={u_stat:.6f}, p={p_val_u:.6g}')
		except Exception:
			print('Mann-Whitney U: failed')
	else:
		print('Mann-Whitney U: unavailable')

	d = cohens_d(vals_a, vals_b)
	delta = cliffs_delta(vals_a, vals_b)
	print(f"Cohen's d: {d:.6f}")
	print(f"Cliff's delta: {delta:.6f}")

	if except_available and len(vals_a) > 1:
		se = np.std(vals_a, ddof=1) / math.sqrt(len(vals_a))
		print(f'Group A approx 95% CI: [{mean_a - 1.96*se:.6f}, {mean_a + 1.96*se:.6f}]')
	if except_available and len(vals_b) > 1:
		se = np.std(vals_b, ddof=1) / math.sqrt(len(vals_b))
		print(f'Group B approx 95% CI: [{mean_b - 1.96*se:.6f}, {mean_b + 1.96*se:.6f}]')


if __name__ == '__main__':
	main()
