// corpus.js — default training text for the early lessons.
// Deliberately small and repetitive: an n-gram needs repetition to say anything at all,
// and you can read the whole thing, so you can check the model against its own evidence.

export const DEFAULT_CORPUS = `The engine loads the weights from a file on disk.
The engine reads the prompt and turns the prompt into tokens.
The engine turns the tokens into vectors and the vectors into logits.
The sampler turns the logits into one token.
The server sends the token to the client and the loop runs again.

A token is a small piece of text. A token is not a word and a token is not a letter.
The model does not read text. The model reads numbers.
The weights are numbers. The activations are numbers. The logits are numbers.
Every number lives in memory and every number was read from the file.

The attention layer looks at the earlier tokens.
The attention layer compares the query with every key and scores every key.
The scores become weights and the weights mix the values.
The mixed values go back into the stream and the stream goes into the next layer.

The feed forward layer does not look at other tokens.
The feed forward layer takes one vector and returns one vector.
The gate decides how much of the signal passes.

The cache holds the keys and the values from the earlier tokens.
Without the cache the engine would read the whole prompt again for every token.
With the cache the engine reads the weights once for every token.
Reading the weights takes time and reading the weights takes bandwidth.
The decode step is slow because memory is slow, not because the math is hard.

The quantiser packs eight weights into one word.
The kernel unpacks the weights while the kernel multiplies.
A scale and a bias turn a small integer back into a number near the original number.
The error is small and the error is measured, never assumed.

Temperature changes how sharp the distribution is.
A low temperature makes the model repeat itself.
A high temperature makes the model wander.
Top k keeps the best few tokens and drops the rest.
Top p keeps the smallest set of tokens that covers enough probability.
Greedy decoding always takes the best token and greedy decoding is boring.`;

export const CORPUS_ALT = `the cat sat on the mat
the cat sat on the hat
the dog sat on the mat
the dog ate the cat food
the cat ate the dog food
a cat and a dog sat on a mat
the mat was flat and the hat was flat
the dog was fat and the cat was fat`;
