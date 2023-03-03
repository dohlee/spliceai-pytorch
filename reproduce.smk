models = ['80nt', '400nt', '2k', '10k']
model2l = {'80nt': 80, '400nt': 400, '2k': 2000, '10k': 10000}
seeds = [42, 43, 44, 45, 46]

rule all:
    input: expand('ckpts/{model}_{seed}.pt', model=models, seed=seeds)

rule train:
    input:
        train = lambda wildcards: f'spliceai_train_code/Canonical/dataset_train_all.{model2l[wildcards.model]}.h5',
        test = lambda wildcards: f'spliceai_train_code/Canonical/dataset_test_0.{model2l[wildcards.model]}.h5',
    output:
        'ckpts/{model}_{seed}.pt'
    shell:
        'python -m spliceai_pytorch.train '
        '--model {wildcards.model} '
        '--train-h5 {input.train} '
        '--test-h5 {input.test} '
        '--output {output} '
        '--seed {wildcards.seed} '
        '--use-wandb'

