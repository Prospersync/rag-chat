# Databricks notebook source
# DBTITLE 1,Install langchain
# MAGIC %pip install langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Extract and chunk text from CSV file
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read the text file
df = spark.read.text("abfss://jdrew2@adlsbarbarian2.dfs.core.windows.net/csv/webscrape.csv")

# Collect all the text into a single string
text_column = " ".join([row.value for row in df.collect()])

length_function = len

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=length_function,
)
chunks = splitter.split_text(text_column)

# COMMAND ----------

# DBTITLE 1,Pandas UDF to chunk text data for insert
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StringType
import pandas as pd

@pandas_udf("array<string>")
def get_chunks(dummy):
    return pd.Series([chunks])

# Register the UDF
spark.udf.register("get_chunks_udf", get_chunks)

# COMMAND ----------

# DBTITLE 1,Insert chunked data into docs_text table
# MAGIC %sql
# MAGIC insert into llm.rag.docs_text (text)
# MAGIC select explode(get_chunks_udf('dummy')) as text;
