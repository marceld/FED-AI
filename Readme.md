# Generating Federal Reserve Statements with AI

![ScreenShot](/screenshots/bernanke.png)


This repo contains the code and data to train OpenAI's GPT-2 language model to learn how to create fake monetary policy statements like the ones regularly issued by the Federal Reserve.

The open-source model can also mimic "Fedspeak" when given a few words with which a statement is meant to begin, e.g. "The global financial crisis demands..." and other prefixes.

### Examples:


> "**The global financial crisis demands** a rapid response of monetary policy. The Committee is concerned that this conflict could lead to a continuing deterioration in business conditions that could contribute to inflationary imbalances in the economy that could undermine the favorable performance of the economy and therefore supports the adoption of stringent measures…"

> "**Copper** futures have increased in recent weeks, and some analysts have raised concerns about the potential for inflationary imbalances that could undermine economic growth…"

> "**Global risks** to the economic outlook have shifted in recent months. Inflation and longer-term inflation expectations remain well contained. The Committee perceives the upside and downside risks to the attainment of both sustainable growth and price stability for the next few quarters to be roughly equal…"

### Overview

In order to create AI-generated Fed Statements we need to go through the following steps:
1. Data and tools - scrape and clean statements from the Fed website
2. Quick exploratory data analysis (EDA)
3. Loading and fine-tuning the GPT-2 language model
4. Examine results - generic Fed statements and those with a given prefix


Please have a look at a more detailed description in this [Medium article](https://medium.com/@marceldietsch/how-to-generate-federal-reserve-statements-with-ai-8fe5da3ae5a5)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine (and Google Colab) for development and testing purposes.

### Key prerequisites

```python
# Jupyter
Python==3.6
Pandas==0.24.2
spaCy==2.1
# Colab
GPT_2_simple
```

### Installing & running the code

Run the [Jupyter notebook](https://github.com/marceld/FED-AI/blob/master/Scrape%20and%20clean%20Fed%20statements.ipynb) first to scrape and clean the data

Afterwards use the [Colab notebook](https://github.com/marceld/FED-AI/blob/master/Fedspeak_Fine_tune_openAI's_GPT_2_model_with_Federal_Reserve_statements_of_the_past_25_years.ipynb) (based on [Max Woolf's work](https://github.com/minimaxir/gpt-2-simple)) and install the Python wrapper for the OpenAI model:
```Python
pip install -q gpt-2-simple
```

Select the medium-size (i.e. the largest available) model if you have enough space:
```python
gpt2.download_gpt2(model_name="345M")
```
The clean text data from the scraper should be stored in /data - copy it over to your Google Drive
and feed it to the model.
```python
file_name = "text_clean.txt"
gpt2.copy_file_from_gdrive(file_name)
```

#### Train the model with TensorFlow
```python
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='345M',
              steps=2000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )
```

That's what the fine-tuning should look like:

![ScreenShot](/screenshots/DL_train_10x.gif)


#### Generate Fed statements
```python
gpt2.generate(sess,
              length=250,
              temperature=0.7,
              prefix="The stock market volatility",
              nsamples=5,
              batch_size=5
              )
```



## Example output

This is what the model produced - without a prefix:

> In response to the deterioration in the labor market, the Committee decided to extend the average maturity of its holdings of securities. The Committee will regularly review the size and composition of its securities holdings and is prepared to adjust those holdings as appropriate. The Committee also decided to keep the target range for the federal funds rate at 0 to 1/4 percent and currently anticipates that economic conditions — including low rates of resource utilization and a subdued outlook for inflation over the medium run — are likely to warrant exceptionally low levels of the federal funds rate at least through mid-2013. The Committee will continue to assess the economic outlook in light of incoming information and is prepared to employ its tools to promote a stronger economic recovery in a context of price stability.

Sounds like real statement the Fed could have made.


It sometimes does get things wrong though. If we give it the prefix "President Trump"

> **President Trump** lifted the temporary restrictions placed on certain foreign investors and temporarily suspended the capital gains tax on the U.S. Treasury securities market.[...] In order to promote a smooth transition in markets, the Committee will gradually slow the pace of its purchases of both agency debt and agency mortgage-backed securities and anticipates that these transactions will be executed by the end of the first quarter of **2010**.

The statement above starts with the highlighted prefix but the paragraph ends with 2010! The issue is obvious: Obama was president in 2010, not Trump. My guess is that the model learnt that Obama was president and it also knows that Trump is now president and hence might have swapped the names given that the context is similar.


## Author

* **Marcel Dietsch** - (https://github.com/marceld) and (https://twitter.com/MarcelDietsch)

## License

This project is licensed under the MIT License
