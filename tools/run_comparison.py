import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from compare.data import (get_device, load_mnist_arrays, load_mnist_loaders,
                          load_synthetic_dataset, seed_everything)
from compare.models import DEEP_MODELS, count_params
from compare.train_utils import plot_curves, train_deep
from compare.traditional import run_traditional_all
from compare import ddpm as ddpm_lib

DEEP_NAMES = ['ANN', 'CNN', 'ViT']


def parse_args():
    p = argparse.ArgumentParser(
        description='Unified MNIST comparison: ANN / CNN / ViT / traditional ML, '
                    'plus classifiers trained on DDPM-generated data')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-synthetic', type=int, default=2000)
    p.add_argument('--gen-batch', type=int, default=250)
    p.add_argument('--confidence', type=float, default=0.9)
    p.add_argument('--out', type=str, default='compare_results')
    p.add_argument('--config', type=str, default='config/default.yaml')
    p.add_argument('--ckpt', type=str, default='default/ddpm_ckpt.pth')
    p.add_argument('--skip-generation', action='store_true',
                   help='reuse existing ddpm_synthetic.pt if present')
    p.add_argument('--skip-traditional', action='store_true')
    p.add_argument('--with-augment', action='store_true',
                   help='also train deep models on real+synthetic combined data')
    p.add_argument('--smoke', action='store_true',
                   help='tiny fast run to verify the pipeline end to end')
    return p.parse_args()


def add_row(results, part, model, train_source, train_size, params, time_s, best_acc, final_acc):
    results.append({
        'part': part,
        'model': model,
        'train_source': train_source,
        'train_size': train_size,
        'params': params,
        'time_s': round(time_s, 1) if time_s is not None else '',
        'best_acc': round(best_acc, 4) if best_acc is not None else '',
        'final_acc': round(final_acc, 4) if final_acc is not None else '',
    })


def run_deep_experiments(results, names, train_loader, test_loader, source,
                         args, device, curves_dir, tag):
    trained = {}
    for name in names:
        seed_everything(args.seed)
        model = DEEP_MODELS[name]()
        trained_model, hist, elapsed = train_deep(
            model, train_loader, test_loader, device,
            num_epochs=args.epochs, lr=args.lr, desc=f'{name}-{tag}')
        best = max(hist['acc'])
        add_row(results, tag, name, source, len(train_loader.dataset),
                count_params(model), elapsed, best, hist['acc'][-1])
        plot_curves(hist, f'{name} ({source}, {args.epochs} epochs)',
                    os.path.join(curves_dir, f'{name}_{tag}.png'))
        trained[name] = {'model': trained_model, 'best_acc': best}
        print(f'[Done] {name}-{tag}: best acc {best:.4f}, {elapsed:.1f}s')
    return trained


def run_traditional(results, args):
    x_train, y_train, x_test, y_test = load_mnist_arrays()
    res = run_traditional_all(x_train, y_train, x_test, y_test)
    for name, r in res.items():
        add_row(results, 'traditional', name, 'real', len(y_train), '-', r['time'],
                r['acc'], r['acc'])


def ensure_synthetic(results, args, device, trained_real):
    synth_path = os.path.join(args.out, 'ddpm_synthetic.pt')
    meta = None
    if args.skip_generation and os.path.exists(synth_path):
        synth_ds, meta = load_synthetic_dataset(synth_path)
        print(f'[Info] Reusing synthetic dataset: {len(synth_ds)} samples from {synth_path}')
        return synth_ds, meta

    ddpm_model, scheduler, diffusion_config, model_config = ddpm_lib.load_ddpm(
        args.config, args.ckpt, device)
    images = ddpm_lib.generate_samples(
        ddpm_model, scheduler, diffusion_config, model_config,
        args.num_synthetic, device, gen_batch=args.gen_batch, seed=args.seed)
    ddpm_lib.save_preview_grid(images, os.path.join(args.out, 'ddpm_preview.png'))

    labeler_name = max(trained_real, key=lambda k: trained_real[k]['best_acc'])
    labeler = trained_real[labeler_name]['model']
    labels_all, confs_all, keep = ddpm_lib.pseudo_label(
        images, labeler, device, threshold=args.confidence)

    if int(keep.sum()) == 0:
        print('[Warn] No sample passed the confidence threshold; '
              'falling back to raw argmax pseudo-labels.')
        kept_images, kept_labels = images, labels_all
        fallback = True
    else:
        kept_images, kept_labels = images[keep], labels_all[keep]
        fallback = False

    meta = {
        'ckpt': args.ckpt,
        'num_timesteps': diffusion_config['num_timesteps'],
        'num_generated': int(args.num_synthetic),
        'pseudo_labeler': labeler_name,
        'labeler_real_best_acc': round(trained_real[labeler_name]['best_acc'], 4),
        'confidence_threshold': args.confidence,
        'threshold_fallback': fallback,
        'num_kept': int(len(kept_labels)),
        'kept_ratio': round(float(len(kept_labels)) / max(len(images), 1), 4),
    }
    torch.save({'images': kept_images.float(), 'labels': kept_labels.long(), 'meta': meta},
               synth_path)
    print(f'[Info] Synthetic dataset saved: {synth_path} ({len(kept_labels)} kept, '
          f'labeler={labeler_name})')
    return TensorDataset(kept_images.float(), kept_labels.long()), meta


def format_params(p):
    if p in ('-', ''):
        return p
    return f'{int(p):,}'


def write_summary(results, meta, args, device, out_md, out_csv):
    protocol = [
        f'- Device: {device}',
        f'- Seed: {args.seed}, Epochs: {args.epochs}, Batch size: {args.batch_size}, '
        f'Optimizer: Adam(lr={args.lr}) + StepLR(step=5, gamma=0.5), Loss: CrossEntropy',
        '- Data: official MNIST (60k train / 10k test), Normalize((0.5,), (0.5,)) -> [-1,1], '
        'num_workers=0',
        '- Traditional ML inputs: flattened 784-dim vectors scaled to [-1,1]',
    ]
    if meta is not None:
        protocol += [
            f'- DDPM checkpoint: {meta["ckpt"]}, sampling timesteps: {meta["num_timesteps"]}, '
            f'generated: {meta["num_generated"]}',
            f'- Synthetic labels: pseudo-labels from {meta["pseudo_labeler"]} '
            f'(its real-data best acc {meta["labeler_real_best_acc"]}), '
            f'confidence >= {meta["confidence_threshold"]}, '
            f'kept {meta["num_kept"]}/{meta["num_generated"]} '
            f'({"threshold fallback, all kept" if meta["threshold_fallback"] else "filtered"})',
        ]
    if args.smoke:
        protocol.append('- NOTE: smoke mode, results are meaningless, pipeline check only')

    order = {'real': 0, 'traditional': 1, 'ddpm_synthetic': 2, 'real+ddpm_synthetic': 3}
    groups = {}
    for row in results:
        groups.setdefault(row['train_source'], []).append(row)

    lines = ['# MNIST Comparison Results', '', '## Protocol', ''] + protocol + ['']
    for source in sorted(groups, key=lambda s: order.get(s, 9)):
        lines.append(f'## Trained on: {source}')
        lines.append('')
        lines.append('| Model | Train size | Params | Time (s) | Best Acc | Final Acc |')
        lines.append('|---|---|---|---|---|---|')
        for row in groups[source]:
            lines.append('| {model} | {train_size} | {params} | {time_s} | {best_acc} | {final_acc} |'.format(
                model=row['model'], train_size=row['train_size'],
                params=format_params(row['params']), time_s=row['time_s'],
                best_acc=row['best_acc'], final_acc=row['final_acc']))
        lines.append('')

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    fieldnames = ['part', 'model', 'train_source', 'train_size', 'params', 'time_s',
                  'best_acc', 'final_acc']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print('\n===== Summary =====')
    for line in lines:
        if line.startswith('|') or line.startswith('##'):
            print(line)
    print(f'\n[Saved] {out_md}')
    print(f'[Saved] {out_csv}')


def main():
    args = parse_args()
    limit_train = limit_test = gen_timesteps = None
    if args.smoke:
        args.epochs = 1
        args.num_synthetic = 12
        args.gen_batch = 12
        args.skip_traditional = True
        limit_train, limit_test, gen_timesteps = 600, 1024, 100
        print('[Info] Smoke mode enabled: tiny subset, 1 epoch, 100 sampling steps')

    seed_everything(args.seed)
    device = get_device()
    print(f'[Info] Device: {device}')
    if str(device) == 'cpu' and args.num_synthetic > 1000:
        print(f'[Warn] Running on CPU with num-synthetic={args.num_synthetic}; '
              'DDPM sampling may take very long. Consider a smaller value or a CUDA env.')

    os.makedirs(args.out, exist_ok=True)
    curves_dir = os.path.join(args.out, 'curves')
    os.makedirs(curves_dir, exist_ok=True)

    real_train_loader, real_test_loader, real_train_ds, _ = load_mnist_loaders(
        args.batch_size, limit_train=limit_train, limit_test=limit_test)

    results = []
    trained_real = run_deep_experiments(
        results, DEEP_NAMES, real_train_loader, real_test_loader,
        'real', args, device, curves_dir, 'real')

    if not args.skip_traditional:
        run_traditional(results, args)

    synth_ds, meta = ensure_synthetic(results, args, device, trained_real)
    if len(synth_ds) == 0:
        print('[Error] Synthetic dataset is empty; aborting DDPM-based experiments.')
        write_summary(results, meta, args, device,
                      os.path.join(args.out, 'summary.md'),
                      os.path.join(args.out, 'summary.csv'))
        return

    synth_loader = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    run_deep_experiments(results, DEEP_NAMES, synth_loader, real_test_loader,
                         'ddpm_synthetic', args, device, curves_dir, 'ddpm')

    if args.with_augment:
        aug_ds = ConcatDataset([real_train_ds, synth_ds])
        aug_loader = DataLoader(aug_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        run_deep_experiments(results, DEEP_NAMES, aug_loader, real_test_loader,
                             'real+ddpm_synthetic', args, device, curves_dir, 'augment')

    write_summary(results, meta, args, device,
                  os.path.join(args.out, 'summary.md'),
                  os.path.join(args.out, 'summary.csv'))
    print('[Done] All comparisons finished.')


if __name__ == '__main__':
    main()
