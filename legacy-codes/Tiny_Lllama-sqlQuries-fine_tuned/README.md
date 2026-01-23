# Tiny_Lllama-sqlQuries-fine_tuned
Fine Tuned TinyLlama 1.1 b on sql-create-context

Model Name : TinyLlama/TinyLlama-1.1B-Chat-v1.0

Dataset : b-mc2/sql-create-context

Requirements : ! pip install torch transformers trl accelerate peft datasets bitsandbytes pandas

Final model : siddharth-magesh/Tiny_Lllama-sqlQuries-fine_tuned

POC deployed on streamLit

input fields : {context : table creation query , input : question }
output : {Following query}

Things to be done :

---training epochs should be increased
---database should be fetched to the model directly 
---database structure should be automatically sent to the model
---the out tokens should be string formatted and fetch to the table to get the actual values
