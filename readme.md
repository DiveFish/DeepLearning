# Named Entity Recognition

Use a bidirectional LSTM in combination with a conditional random field layer in order to correctly tag German sentences with namend-entity tags.

## The tagset
### Token tags
B: first named-entity token<br />
I: inside of named entity<br />
O: other token

### Named entity descriptions
LOC: location<br />
ORG: organization<br />
PER: person<br />
OTH: other

### Further specifications
deriv: token derived from a name<br />
part: part of token is a name

```
1	Das       O
2       britische       B-LOCderiv
3       Label       O
4       EMI       B-OR
```
