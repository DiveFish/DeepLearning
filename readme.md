# Named Entity Recognition

This program uses a bidirectional LSTM in combination with a conditional random field layer in order to tag German sentences with namend-entity tags.

## The tagset
### Token tags
**B**: first named-entity token<br />
**I**: inside of named entity<br />
**O**: other token

### Named entity descriptions
**LOC**: location<br />
**ORG**: organization<br />
**PER**: person<br />
**OTH**: other

### Further specifications
**deriv**: token derived from a name<br />
**part**: part of token is a name

## Examples
```
1	Das       O
2       britische       B-LOCderiv
3       Label       O
4       EMI       B-OR
```
# Usage
For running the script you need to specify a directory containing files in .conll format which contain the input data. Training and Test data will be splitted automaticall. 
As a second argument you need to specify a file containing pretrained word embeddings.
```
python3 train.py path/to/directory/withConllFiles path/to/word/embeddings
```
