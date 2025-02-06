### **Rohdaten**  
Ursprünglich wollte ich nicht nur eine maschinelle Lernaufgabe und ein Benchmarking abschließen, sondern auch die Gehaltsunterschiede zwischen der KI-Branche und traditionellen Branchen detailliert analysieren. Dazu wollte ich KI-Gehälter nach Ländern, Geschlecht, Berufserfahrung usw. untersuchen, um das Potenzial der KI-Branche zu bewerten.  
Leider war ich nicht in der Lage, meinen ursprünglichen Plan vollständig umzusetzen.  
Ich habe viele Datensätze gesammelt, die in der letzten Sektion aufgeführt sind.

**Die für die Vorhersage verwendeten Datensätze nach der Auswahl:**  
1. `cleaned_global-salaries-in-ai-ml-data-science.csv`: *(work_year, experience_level, employment_type, US, company_size, remote_ratio) → salary_in_usd*  
2. `cleaned_global_ai_ml_data_salaries.csv`: *(work_year, experience_level, employment_type, remote_ratio, company_size) → salary_in_usd*  
3. `cleaned_jobs_in_data_2024.csv`: *(work_year, experience_level, employment_type, company_size) → SalaryUSD*  
4. `cleaned_data_science_salaries.csv`: *(work_year, experience_level, employment_type, work_models=remote_ratio, company_size) → salary_in_usd*  

**Die ursprünglich für den Vergleich verschiedener Branchen geplanten Datensätze (aufgrund von Zeitmangel verworfen):**  
1. `cleaned_Salary.csv`  
2. `cleaned_Salary_Data_Based_country_and_race.csv`  
3. `cleaned_Salary_Data_2022_REV15.csv`  
4. `cleaned_global_jobs_salaries_2024.csv`  
5. `cleaned_US_Data_Jobs_Salaries_Dataset.csv`  
6. `cleaned_combined_salaries.csv`  

---

### **Datenbereinigung**  
Nachdem ich die Datensätze erhalten hatte, versuchte ich zunächst, in `scripts` eine Übersicht über alle Datensätze zu erstellen.  
Da das Skript jedoch nicht gut funktionierte, änderte ich meine Strategie und erstellte in jedem `raw_data`-Ordner ein **Jupyter-Notebook**, um die **Bereinigung und Standardisierung** einzeln durchzuführen.  

---

### **Modelltraining und Validierung**  
- `new_prediction.ipynb`: Verwendet vollständig **SKlearn**-Methoden.  
- `spark_train.py`: Implementiert die **PySpark**-Methode.  

---

### **Starten aller Hadoop-Dienste**  
```bash
conda activate hadoop-ml
start-dfs.sh
start-yarn.sh
mapred --daemon start historyserver
```
### Hochladen von Datensätzen und Skripten
```bash
scp -r "G:\Nextcloud\FSU_Cloud\Big Data\Projekt\cleaned_data" hadoop@192.168.88.101:/home/hadoop/
```
### Hochladen in HDFS:
```bash
hdfs dfs -put /home/hadoop/cleaned_data /user/hadoop/datasets/
hdfs dfs -ls /user/hadoop/datasets/cleaned_data
```

### Skript-Ausführung mit spark-submit
```bash
spark-submit --master yarn \
  --deploy-mode cluster \
  --conf spark.scheduler.mode=FAIR \
  --conf spark.pyspark.python=/home/hadoop/miniconda3/envs/hadoop-ml/bin/python3.8 \
  --conf spark.pyspark.driver.python=/home/hadoop/miniconda3/envs/hadoop-ml/bin/python3.8 \
  /home/hadoop/scripts/spark_train.py
```
Ich musste diesen Schritt mehrmals testen, da verschiedene Conda- und Spark-Umgebungsprobleme auftraten.
Das endgültig funktionierende spark-submit-Format ist oben dargestellt.
### Ergebnisse zurück auf das lokale System übertragen

```bash
scp -r hadoop@192.168.88.101:/home/hadoop/results/* "G:\Nextcloud\FSU_Cloud\Big Data\Projekt\results"

```
### Finale Benchmark-Ergebnisse
```bash
SKlearn
RandomForest: 1 cpu  Total Execution Time: 283.2614 seconds
              3 cpu  Total Training & Prediction Time: 118.3298 seconds

linear Regression: 1 cpu: 1 cpu Total Training & Prediction Time: 5.5920 seconds
                   3 cpu: 3 cpus Total Training & Prediction Time: 5.5880 seconds

pyspark

Random Forest - (Time: 30.77s)
Linear Regression -  (Time: 3.92s)
```

### Beenden aller Hadoop-Dienste
```bash
stop-dfs.sh
stop-yarn.sh
mapred --daemon stop historyserver
```
### Dataset-Download

---
[Global AI, ML, and Data Science Salaries](https://www.kaggle.com/datasets/msjahid/global-ai-ml-and-data-science-salaries)
[Global Salaries in AI, ML, Data Science](https://www.kaggle.com/datasets/aijobs/global-salaries-in-ai-ml-data-science?select=salaries.csv)
[EDA: Salary Data Science (2020-2024)](https://www.kaggle.com/code/gabrielfelinto/eda-salary-data-science-2020-2024)

🔗 [Jobs and Salaries in Data Field 2024](https://www.kaggle.com/datasets/murilozangari/jobs-and-salaries-in-data-field-2024)


[Salary by Job Title and Country](https://www.kaggle.com/datasets/amirmahdiabbootalebi/salary-by-job-title-and-country)



📊 [From Data Entry to CEO: The AI Job Threat Index](https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index)

🤖 [Industrial Robotics and Automation Dataset](https://www.kaggle.com/datasets/kennedywanakacha/industrial-robotics-and-automation-dataset)

📊 [AI-Powered Job Market Insights](https://www.kaggle.com/datasets/uom190346a/ai-powered-job-market-insights)

💻 [Software Professional Salaries 2022](https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022)



[Data Jobs Dataset (Kaggle)](https://www.kaggle.com/datasets/juanmerinobermejo/data-jobs-dataset?select=jobs.csv)


[Global Salary DataSet 2022](https://www.kaggle.com/datasets/ricardoaugas/salary-transparency-dataset-2022)





```linux
kaggle datasets download -d murilozangari/jobs-and-salaries-in-data-field-2024

kaggle datasets download -d manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index

kaggle datasets download -d kennedywanakacha/industrial-robotics-and-automation-dataset

kaggle datasets download -d andrewmvd/occupation-salary-and-likelihood-of-automation

kaggle datasets download -d iamsouravbanerjee/software-professional-salaries-2022

kaggle datasets download -d sazidthe1/data-science-salaries

kaggle datasets download -d uom190346a/ai-powered-job-market-insights

kaggle datasets download -d juanmerinobermejo/data-jobs-dataset

kaggle datasets download -d ronaldonyango/global-jobs-and-salaries-2024

kaggle datasets download -d aijobs/global-salaries-in-ai-ml-data-science

kaggle datasets download -d ricardoaugas/salary-transparency-dataset-2022

kaggle datasets download -d msjahid/global-ai-ml-and-data-science-salaries

kaggle datasets download -d amirmahdiabbootalebi/salary-by-job-title-and-country

```














