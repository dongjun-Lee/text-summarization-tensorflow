# tensorflow-text-summarization
Simple Tensorflow implementation of text summarization using [seq2seq library](https://www.tensorflow.org/api_guides/python/contrib.seq2seq).


## Model
Encoder-Decoder model with attention mechanism.

### Word Embedding
Used [Glove pre-trained vectors](https://nlp.stanford.edu/projects/glove/) to initialize word embedding.

### Encoder
Used LSTM cell with [stack_bidirectional_dynamic_rnn](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn).

### Decoder
Used LSTM [BasicDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder) for training, and [BeamSearchDecoder](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BeamSearchDecoder) for inference.

### Attention Mechanism
Used [BahdanauAttention](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BahdanauAttention) with weight normalization.


## Requirements
- Python 3
- Tensorflow (>=1.8.0)
- pip install -r requirements.txt


## Usage
### Prepare data
Dataset is available at [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary). Locate the summary.tar.gz file in project root directory. Then,
```
$ python prep_data.py
```
To use Glove pre-trained embedding, download it via
```
$ python prep_data.py --glove
```

### Train
We used ```sumdata/train/train.article.txt``` and ```sumdata/train/train.title.txt``` for training data. To train the model, use
```
$ python train.py
```
To use Glove pre-trained vectors as initial embedding, use
```
$ python train.py --glove
```

#### Additional Hyperparamters
```
$ python train.py -h
usage: train.py [-h] [--num_hidden NUM_HIDDEN] [--num_layers NUM_LAYERS]
                [--beam_width BEAM_WIDTH] [--glove]
                [--embedding_size EMBEDDING_SIZE]
                [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--keep_prob KEEP_PROB] [--toy]

optional arguments:
  -h, --help            show this help message and exit
  --num_hidden NUM_HIDDEN
                        Network size.
  --num_layers NUM_LAYERS
                        Network depth.
  --beam_width BEAM_WIDTH
                        Beam width for beam search decoder.
  --glove               Use glove as initial word embedding.
  --embedding_size EMBEDDING_SIZE
                        Word embedding size.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_epochs NUM_EPOCHS
                        Number of epochs.
  --keep_prob KEEP_PROB
                        Dropout keep prob.
  --toy                 Use only 5K samples of data

```


### Test
Generate summary of each article in ```sumdata/train/valid.article.filter.txt``` by
```
$ python test.py
```
It will generate result summary file ```result.txt```. Check out ROUGE metrics between ```result.txt``` and ```sumdata/train/valid.title.filter.txt``` using [pltrdy/files2rouge](https://github.com/pltrdy/files2rouge).

#### Sample Summary Output
```
"general motors corp. said wednesday its us sales fell ##.# percent in december and four percent in #### with the biggest losses coming from passenger car sales ."
> Model output: gm us sales down # percent in december
> Actual title: gm december sales fall # percent

"japanese share prices rose #.## percent thursday to <unk> highest closing high for more than five years as fresh gains on wall street fanned upbeat investor sentiment , dealers said ."
> Model output:  tokyo shares close # percent higher
> Actual title: tokyo shares close up # percent

"hong kong share prices opened #.## percent higher thursday on follow-through interest in properties after wednesday 's sharp gains on abating interest rate worries , dealers said ."
> Model output: hong kong shares open higher
> Actual title: hong kong shares open higher as rate worries ease

"the dollar regained some lost ground in asian trade thursday in what was seen as a largely technical rebound after weakness prompted by expectations of a shift in us interest rate policy , dealers said ."
> Model output: dollar stable in asian trade
> Actual title: dollar regains ground in asian trade

"the final results of iraq 's december general elections are due within the next four days , a member of the iraqi electoral commission said on thursday ."
> Model output: iraqi election results due in next four days
> Actual title: iraqi election final results out within four days

"microsoft chairman bill gates late wednesday unveiled his vision of the digital lifestyle , outlining the latest version of his windows operating system to be launched later this year ."
> Model output: bill gates unveils new technology vision
> Actual title: gates unveils microsoft 's vision of digital lifestyle
```

## Pre-trained Model
To test with pre-trained model, download [pre_trained.zip](https://drive.google.com/file/d/1V8pS1eoiv51wfiVp2rOB7IvJ5PeQs2n-/view?usp=sharing), and locate it in the project root directory. Then,
```
$ unzip pre_trained.zip
$ python test.py
```
