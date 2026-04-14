#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 11pt,
)
#set heading(numbering: "1.")

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

== Transformer model