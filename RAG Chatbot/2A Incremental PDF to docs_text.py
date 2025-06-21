# Databricks notebook source
# DBTITLE 1,Install Pdfplumber and langchain
# MAGIC %pip install Pdfplumber langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,List pdf files in Volume
import os
from pyspark.sql.functions import substring_index

# Directory path
directory_path = "/Volumes/workspace/chatbot/source_docs"

# List files in directory
file_paths = [file.path for file in dbutils.fs.ls(directory_path)]

# Extract file names from paths
df = spark.createDataFrame(file_paths, "string").select(substring_index("value", "/", -1).alias("file_name"))

# Show dataframe
df.show()

# COMMAND ----------

# DBTITLE 1,Check for files not yet processed, extract text and chunk
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_volume_path = "/Volumes/agents/irpf/source_docs"  # Specify the path to the PDF volume directory

# Get the list of already processed PDF files from the Delta table
processed_files = spark.sql(f"SELECT DISTINCT file_name FROM agents.irpf.docs_track").collect()
processed_files = set(row["file_name"] for row in processed_files)

# Process only new PDF files
new_files = [file for file in os.listdir(pdf_volume_path) if file not in processed_files]

all_text = ''  # Initialize all_text to store text from new PDF files

for file_name in new_files:
    # Extract text from the PDF file
    pdf_path = os.path.join(pdf_volume_path, file_name)

    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in pdf.pages:
            single_page_text = pdf_page.extract_text()
            # Separate each page's text with newline
            all_text = all_text + '\n' + single_page_text

# Split the combined text into chunks using the RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

length_function = len

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=length_function,
)
chunks = splitter.split_text(all_text)

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
# MAGIC insert into agents.irpf.docs_text (text)
# MAGIC select explode(get_chunks_udf('dummy')) as text;

# COMMAND ----------

# DBTITLE 1,Update docs_track table so same files aren't processed again
df.createOrReplaceTempView("temp_table")  # Create a temporary table from the DataFrame

# Insert only the rows that do not exist in the target table
spark.sql("""
    INSERT INTO agents.irpf.docs_track
    SELECT * FROM temp_table
    WHERE NOT EXISTS (
        SELECT 1 FROM agents.irpf.docs_track
        WHERE temp_table.file_name = agents.irpf.docs_track.file_name
    )
""")
