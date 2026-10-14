"""Train NAG-MVC and cluster the final representations using the configured K."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import numpy as np
from sklearn.cluster import KMeans
from nag_mvc.features import prepare_features, compute_subject_discordance, assemble_representation
from nag_mvc.training import _train_seed


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=Path)
    group.add_argument('--demo', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='')
    args = parser.parse_args()
    cfg = {
        'hidden_dim': 256,
        'shared_dim': 64,
        'private_dim': 32,
        'proj_dim': 64,
        'epochs': 100,
        'batch_size': 256,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'loss_reconstruction': 1.0,
        'loss_contrast': 1.0,
        'loss_orthogonality': 0.05,
        'loss_redundancy': 0.1,
        'n_neighbors': 50,
        'reliability_temperature': 0.2,
        'averaging_strength': 0.2,
        'seeds': [42, 202, 1002],
        'selected_k': 2,
    }
    if args.demo:
        rng = np.random.default_rng(11)
        local = rng.normal(size=(64,4,116)).astype(np.float32)
        sites = np.asarray([str(i%4) for i in range(64)])
        cfg.update(epochs=2,batch_size=32,seeds=[42])
    else:
        with np.load(args.dataset, allow_pickle=False) as data:
            expected = ['ALFF','fALFF','ReHo','DC']
            views = data['view_names'].astype(str).tolist()
            if views != expected:
                raise ValueError(f'Expected view_names={expected}; got {views}')
            mask = data['groups'].astype(str) == '1'
            local = data['local_metrics'][mask].astype(np.float32)
            sites = data['sites'][mask].astype(str)
    x = prepare_features(local, sites)
    q, fusion, _ = compute_subject_discordance(x,cfg['n_neighbors'],cfg['reliability_temperature'])
    args.out_dir.mkdir(parents=True,exist_ok=False)
    (args.out_dir/'run_config.json').write_text(json.dumps(cfg,indent=2))
    training_args = Namespace(**cfg,device=args.device)
    for seed in cfg['seeds']:
        out = args.out_dir/f'seed_{seed}'
        out.mkdir()
        fused = _train_seed(x,q,fusion,training_args,seed,out)
        joint = assemble_representation(fused,x,q,cfg['n_neighbors'],cfg['averaging_strength'])
        labels = KMeans(n_clusters=cfg['selected_k'],n_init=50,random_state=seed).fit_predict(joint)
        np.savez_compressed(out/'latent.npz',fused=fused,joint=joint)
        np.save(out/'labels.npy',labels,allow_pickle=False)
        print(f'seed={seed}, patients={len(x)}, representation={joint.shape}, finished',flush=True)


if __name__ == '__main__':
    main()
