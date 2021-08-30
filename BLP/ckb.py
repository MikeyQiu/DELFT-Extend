from ckb import compose
from ckb import datasets
from ckb import evaluation
from ckb import losses
from ckb import models
from ckb import sampling
from ckb import scoring

from transformers import BertTokenizer
from transformers import BertModel

import torch

_ = torch.manual_seed(42)

device = 'cpu' #  You should own a GPU, it is very slow with cpu.

# Train, valid and test sets are a list of triples.
train = [
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Brown  Sugar'),
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Oil'),
    ('My Favorite Carrot Cake Recipe', 'made_with', 'Applesauce'),

    ('Classic Cheesecake Recipe', 'made_with', 'Block cream cheese'),
    ('Classic Cheesecake Recipe', 'made_with', 'Sugar'),
    ('Classic Cheesecake Recipe', 'made_with', 'Sour cream'),
]

valid = [
    ('My Favorite Carrot Cake Recipe', 'made_with', 'A bit of sugar'),
    ('Classic Cheesecake Recipe', 'made_with', 'Eggs')
]

test = [
    ('My Favorite Strawberry Cake Recipe', 'made_with', 'Fresh Strawberry')
]

# Initialize the dataset, batch size should be small to avoid RAM exceed.
dataset = datasets.Dataset(
    batch_size = 1,
    train = train,
    valid = valid,
    test = test,
    seed = 42,
)

model = models.Transformer(
    model = BertModel.from_pretrained('bert-base-uncased'),
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'),
    entities = dataset.entities,
    relations = dataset.relations,
    gamma = 9,
    scoring = scoring.TransE(),
    device = device,
)

model = model.to(device)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr = 0.00005,
)

evaluation = evaluation.Evaluation(
    entities = dataset.entities,
    relations = dataset.relations,
    true_triples = dataset.train + dataset.valid + dataset.test,
    batch_size = 1,
    device = device,
)

# Number of negative samples to show to the model for each batch.
# Should be small to avoid memory error.
sampling = sampling.NegativeSampling(
    size = 1,
    entities = dataset.entities,
    relations = dataset.relations,
    train_triples = dataset.train,
)

pipeline = compose.Pipeline(
    epochs = 20,
    eval_every = 3, # Eval the model every {eval_every} epochs.
    early_stopping_rounds = 1,
    device = device,
)

pipeline = pipeline.learn(
    model = model,
    dataset = dataset,
    evaluation = evaluation,
    sampling = sampling,
    optimizer = optimizer,
    loss = losses.Adversarial(alpha=0.5),
)
