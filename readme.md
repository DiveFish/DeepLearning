# Named Entity Recognition

Use a bidirectional LSTM in combination with a conditional random field layer in order to correctly tag German sentences with namend-entity tags.

## The tagset
### Token tags
B: first named-entity token
I: inside of named entity
O: other token

### Named entity descriptions
LOC: location
ORG: organization
PER: person
OTH: other

###
deriv: token derived from a name
part: part of token is a name

'''
1	Das       O
2       britische       B-LOCderiv
3       Label       O
4       EMI       B-OR
'''
