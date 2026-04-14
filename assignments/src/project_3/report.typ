#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 11pt,
)
#set heading(numbering: "1.")
#show link: underline
#show table.cell.where(y: 0): strong


= INF265: Project 3: Transformers

== Part 1: Encoder-only model for Movie Review Sentiment Classification

In this task we ...

=== Pre-processing / tokenization

In order to use the data for this task, some pre-processing was required.
Initially, we removed html tags, special characters, and made text lowercase.
This was done to help the model learn. By lowering the amount of different characters,
the model has to learn understanding between less tokens. If we were to keep the upper and lowercase of many words, we would
essentially be doubling the different types of tokens the model could encounter.

After making the data less complex, we tokenized the text. Firstly, we add the `[CLS]`
"start"-token. Then we run it through our tokenizer. Now we have turned our text into tokens.
The model expects inputs to be of a certain size, so we truncate the ids to a set `max_length`, or adding
padding to `max_length` if the text is too short.

== Transformer model (encoding block)

The encoder block consists of two main layers.

Layer 1 includes normalization, followed by multi-head attention, and then dropout.

`Layer Norm` -> `Multi-Head attention` -> `Dropout`

The magic in this layer is what happens in the `Multi-head attention`.
This part linearly projects queries, keys and values some arbitrary `n` times. Attention is performed on each of these projections, which are then added together, and projected a final time.
If one were to use only one head for attention, it would be harder for the model to remember multiple parts of the input text simulatenously.
The idea is that each of the heads can remember information from different represenations.

The attention mechanism works by mapping a query and key-value pairs to some output. Each value is assigned a weight, computed by a function between the query and the key. For scaled dot-product attention, the dot product of the query with all the keys is taken, scaled and ran through a softmax function. The softmax function provides the weights for each value.

Between layer 1 and 2, a skip connection is used, adding the output from layer 1, and the initial input together, before feeding this new output into layer 2.

Layer 2 includes normalization, an MLP block, and dropout.

`Layer Norm` -> `MLP` -> `Dropout`

The MLP block linearly projects the data up 4 times its original dimensions, before applying the `GELU` activation function, adding non-linearity. `GELU` is chosen because it can handle both positive and slightly negative values. `GELU` may work well in this case, because a typical network with `RELU` can suffer #link("https://chadrick-kwag.net/posts/relu-gelu-swish-mish-activation-function-comparison/")[if many neurons in a network become zero].
The output is scaled back down again, before being output.

Finally, another skip connection is used, adding the output of layer 2(including its skip connection) to the final output.

== Overview to approach for solving the tasks

Read #link("https://arxiv.org/abs/1706.03762")[Attention is all you need], and the chapter on transformers in the #link("udlbook.github.io/udlbook/")[course book]

== Results

#figure(
  caption: "Loss over time",
 table(
    columns: (1fr, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [*Data set*], [*Accuracy*],
    ),
    "Train",
    "0.868",
    "Validation",
    "0.853",
    "Test",
    "0.849"
  )
)

#figure(
  caption: "Final performance on datasets",
 table(
    columns: (1fr, auto),
    inset: 5pt,
    align: horizon,
    table.header(
      [*Data set*], [*Accuracy*],
    ),
    "Train",
    "0.868",
    "Validation",
    "0.853",
    "Test",
    "0.849"
  )
)