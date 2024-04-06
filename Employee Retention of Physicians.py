#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


inpPath = "C:/CarolineZiegler/Studium_DCU/8. Semester/Business Project/Primary Data/"
employee_retention_Df = pd.read_csv(inpPath + "EmployeeRetention.csv", delimiter =  ",", header = 0, encoding="latin1")
employee_retention_Df


# In[3]:


for column_name in employee_retention_Df.columns:
    print(column_name)


# In[4]:


#column cleaning
employee_retention_Df.drop([
    "ID",
    "Startzeit",
    "Fertigstellungszeit",
    "E-Mail",
    "Name",
    "Zeitpunkt der letzten Änderung"
], inplace=True, axis=1)
employee_retention_Df


# In[5]:


employee_retention_Df.drop(employee_retention_Df.columns[0], axis=1, inplace=True)
employee_retention_Df


# In[6]:


employee_retention_Df.columns


# In[7]:


employee_retention_Df.rename(columns = {"Wie alt sind Sie?\n": "Age", 
                        "Welches Geschlecht haben Sie?\n":"Gender", 
                        "Wie lange gehen Sie bereits Ihrem Beruf als Ärztin/Arzt nach?\n": "YearsInMedicalPractice", 
                        "Bei was für einer Einrichtung sind Sie derzeit (schwerpunktmäßig) angestellt?\n":"CurrentEmployerType", 
                        "Wie würden Sie den Standort der Einrichtung, in der sie arbeiten, einordnen? Handelt es sich eher um ein ländliches, städtisches oder vorstädtisches Gebiet?\n":"FacilityLocationType", 
                        "Welche Position haben Sie derzeit inne?      \n":"CurrentPosition", 
                        "Welches Arbeitsmodell verfolgen Sie derzeit?      ":"CurrentWorkModel", 
                        "Wie hoch ist Ihre tatsächliche Wochenarbeitszeit inklusive aller Dienste und Überstunden im Durchschnitt?\n":"AverageWeeklyWorkingHours",
                        "Werden Ihre Überstunden überwiegend vergütet oder mit Freizeit ausgeglichen?      \n": "OvertimeCompensation",
                        "Wie beurteilen Sie Ihre derzeitigen Arbeitsbedingungen?\n":"CurrentWorkConditionsRating",
                        "Wie beurteilen Sie die personelle Besetzung im ärztlichen Dienst in Ihrer Einrichtung?\n": "StaffingLevelRating",
                        "Wie haben sich Ihre Arbeitsbedingungen seit der COVID-19-Pandemie entwickelt?\n": "WorkConditionsPostCOVID",
                        "Wurden seit der Covid-19 Pandemie neue oder zusätzliche Maßnahmen ergriffen, um Ihre Arbeitssituation und -bedingungen zu verbessern (z.B. Angebot von alternativen Arbeitszeitenmodellen, spezielle...": "PostCOVIDImprovementsMade",
                        "Wenn ja, welche Maßnahmen wurden getroffen?\n": "ImprovementMeasures",
                        "Ich empfinde es als angenehm, bei meinem Arbeitgeber zu arbeiten": "EnjoyWorkingHere",
                        "Ich fühle mich mit meinem Arbeitgeber persönlich verbunden\n": "PersonalConnectionWithEmployer",
                        "Ich fände es persönlich schade, wenn die Beschäftigung bei meinem Arbeitgeber beendet werden würde": "RegretIfEmploymentEnded",
                        "Ich kann mich mit meinem Arbeitgeber und den Produkten/Dienstleistungen identifizieren\n      ": "IdentifyWithEmployer",
                        "Meine persönlichen Kontakte zu meinem Arbeitsumfeld sind für mich von Bedeutung\n ": "ImportanceOfWorkRelationships",
                        "In gewisser Weise bindet mich der für einen Wechsel benötigte Zeitaufwand an meinen Arbeitgeber\n": "SwitchingCostsTimeInvestment",
                        "Ich bin auf meinen Arbeitgeber angewiesen, weil es zurzeit keine gleichwertigen Alternativen am Markt gibt": "DependenceOnEmployerForAlternatives",
                        "Ich empfinde eine Bindung an meinen Arbeitgeber, weil bei einem Wechsel der von mir investierte Aufwand an Wert verlieren würde": "LossOfInvestmentUponSwitching",
                        "Ich fühle mich an meinen Arbeitgeber gebunden, weil ein Wechsel mit Wechselkosten einhergehen würde": "SwitchingCostsPerceived",
                        "Es wäre nicht fair, die Beziehung mit meinem Arbeitgeber aufzukündigen, weil er sich stets um mich als Arbeitnehmer bemüht hat": "FairnessInStayingWithEmployer",
                        "Aufgrund der langen Beziehung mit meinem Arbeitgeber fühle ich mich zu einer gewissen Rücksichtnahme verpflichtet": "ObligationDueToLongRelationship",
                        "Ich fühle mich in der Angestelltenbeziehung mit dem Arbeitgeber zur Fairness verpflichtet": "FeelingOfFairnessToEmployer",
                        "Moralische Verpflichtungen gegenüber dem Arbeitgeber spielen für mich auch eine Rolle": "MoralObligationsToEmployer",
                        "Ich bin vertraglich an meinen Arbeitgeber gebunden": "ContractualBinding",
                        "Mein Arbeitsvertrag schafft faire Rahmenbedingungen für meine Arbeit ": "FairWorkContractConditions", 
                        "Mein Arbeitsvertrag stellt eine gerechte monetärere Kompensation sicher":"FairMonetaryCompensation", 
                        "Ich würde gerne Teile meines Arbeitsvertrages anpassen":"DesireToModifyWorkContract"}, inplace = True)
employee_retention_Df


# In[8]:


employee_retention_Df.columns


# In[9]:


employee_retention_Df.columns = employee_retention_Df.columns.str.strip()


# In[10]:


employee_retention_Df.rename(columns={"Meine persönlichen Kontakte zu meinem Arbeitsumfeld sind für mich von Bedeutung": "ImportanceOfWorkRelationships"}, inplace=True)
employee_retention_Df.columns


# In[11]:


employee_retention_Df.columns.values[18] = 'ImportanceOfWorkRelationships'
employee_retention_Df.columns


# In[12]:


#understanding the dataset structure and input
round(employee_retention_Df.describe(),2)


# In[13]:


#checking for null values
employee_retention_Df.isnull().sum()


# In[14]:


print(employee_retention_Df.dtypes)


# In[15]:


employee_retention_Df['Age'].unique()


# In[16]:


employee_retention_Df["Age"].replace("666", "66", inplace = True)
employee_retention_Df["Age"].unique()


# In[17]:


employee_retention_Df["Age"].replace("61 Jahre", "61", inplace = True)
employee_retention_Df["Age"].unique()


# In[18]:


employee_retention_Df["Age"].replace("54 J", "54", inplace = True)
employee_retention_Df["Age"].unique()


# In[19]:


employee_retention_Df['Age'] = employee_retention_Df['Age'].astype('int64')


# In[20]:


employee_retention_Df['Gender'].unique()


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


#encoding of the categorical column gender with label encoding
LE = LabelEncoder()
encoded_gender = LE.fit_transform(employee_retention_Df['Gender'])
encoded_gender


# In[23]:


employee_retention_Df['Gender_encoded'] = encoded_gender
employee_retention_Df


# In[24]:


employee_retention_Df['YearsInMedicalPractice'].unique()


# In[25]:


#encoding the ordinal categorical column YearsInMedicalPractice
#midpoint of each range (when applicable) and a suitable representative value for open-ended categories will be used

#mapping dictionary
experience_mapping = {
    '< 2 Jahre': 1,
    '2 - 5 Jahre': 3.5,
    '6 - 10 Jahre': 8,
    '11 - 15 Jahre': 13,
    '16 - 20 Jahre': 18,
    '> 20 Jahre': 25
}

#apply the mapping to the column
employee_retention_Df['Experience_encoded'] = employee_retention_Df['YearsInMedicalPractice'].map(experience_mapping)
employee_retention_Df


# In[26]:


employee_retention_Df['CurrentEmployerType'].unique()


# In[27]:


#using one-hot-encoding for the categorical column CurrentEmployerType which has no inherent order
#mapping from German to English
translation_map = {
    'Krankenhaus in privater Trägerschaft': 'Private Hospital',
    'Universitätsklinikum': 'University Hospital',
    'Kirchliches Krankenhaus': 'Church Hospital',
    'Kommunales Krankenhaus': 'Municipal Hospital',
    'Andere stationäre Einrichtung': 'Other Inpatient Facility',
    'Ambulante Einrichtung': 'Outpatient Facility'
}

#applying the map
employee_retention_Df['CurrentEmployerType'] = employee_retention_Df['CurrentEmployerType'].map(translation_map)

#one-hot-encoding
employee_retention_Df = pd.get_dummies(employee_retention_Df, columns=['CurrentEmployerType'])
employee_retention_Df


# In[28]:


employee_retention_Df['FacilityLocationType'].unique()


# In[29]:


#types of areas have a natural order (from most densely populated to least densely populated), an ordinal encoding can be applied 
#encoding the ordinal categorical column FacilityLocationType
location_mapping = {
    'Städtisch (urban)': 3,  
    'Vorstädtisch (suburban)': 2,  
    'Ländlich (rural)': 1  
}

#apply mapping
employee_retention_Df['FacilityLocationType_encoded'] = employee_retention_Df['FacilityLocationType'].map(location_mapping)
employee_retention_Df


# In[30]:


employee_retention_Df['CurrentPosition'].unique()


# In[31]:


#encoding the hierarchy of medical positions ordinally

#mapping dictionary
position_mapping = {
    'Ärztin/Arzt in Weiterbildung': 1,  # Resident/Doctor in training
    'Fachärztin/arzt': 2,  # Specialist/Attending physician
    'Oberärztin/arzt': 3,  # Senior physician/Consultant
    'Stellv. Chefärztin/arzt': 4,  # Deputy head physician
    'Chefärztin/arzt': 5   # Head physician/Chief physician
}

#apply mapping 
employee_retention_Df['CurrentPosition_encoded'] = employee_retention_Df['CurrentPosition'].map(position_mapping)
employee_retention_Df


# In[32]:


employee_retention_Df['CurrentWorkModel'].unique()


# In[33]:


#binary encoding since there are only two categories without a natural order
#mapping dictionary
work_model_mapping = {
    'Vollzeit': 1,  # Full-time
    'Teilzeit': 0   # Part-time
}

#apply mapping to the column
employee_retention_Df['CurrentWorkModel_encoded'] = employee_retention_Df['CurrentWorkModel'].map(work_model_mapping)
employee_retention_Df


# In[34]:


employee_retention_Df['AverageWeeklyWorkingHours'].unique()


# In[35]:


#map ranges to their midpoint values

#mapping dictionary
hours_mapping = {
    '60 - 79 h': 70,
    '40 - 48 h': 44,
    '30 - 39 h': 35,
    '5 - 19 h': 12,
    '49 - 59 h': 54,
    '20 - 29 h': 25,
    '> 80 h': 85
}

#apply mapping to the column
employee_retention_Df['AverageWeeklyWorkingHours_encoded'] = employee_retention_Df['AverageWeeklyWorkingHours'].map(hours_mapping)
employee_retention_Df


# In[36]:


employee_retention_Df['OvertimeCompensation'].unique()


# In[37]:


#using one-hot-encoding for the categorical column OvertimeCompensation which has no inherent order
#mapping from German to English
translation_map2 = {
    'Überwiegend Freizeitausgleich': 'Mostly Time Off',
    'Weder noch': 'Neither',
    'Überwiegend vergütet': 'Mostly Paid'
}

#applying the map
employee_retention_Df['OvertimeCompensation'] = employee_retention_Df['OvertimeCompensation'].map(translation_map2)

#one-hot-encoding
employee_retention_Df = pd.get_dummies(employee_retention_Df, columns=['OvertimeCompensation'])
employee_retention_Df


# In[38]:


employee_retention_Df['CurrentWorkConditionsRating'].unique()


# In[39]:


#an ordinal encoding can be applied since these categories have a natural order from positive to negative

#mapping dictionary
rating_mapping = {
    'Sehr gut': 4,
    'Gut': 3,
    'Mittelmäßig': 2,
    'Schlecht': 1
}

#apply mapping to the column
employee_retention_Df['CurrentWorkConditionsRating_encoded'] = employee_retention_Df['CurrentWorkConditionsRating'].map(rating_mapping)
employee_retention_Df


# In[40]:


employee_retention_Df['StaffingLevelRating'].unique()


# In[41]:


#an ordinal encoding can be applied since these categories have a natural order from positive to negative

#apply mapping to the column
employee_retention_Df['StaffingLevelRating_encoded'] = employee_retention_Df['StaffingLevelRating'].map(rating_mapping)
employee_retention_Df


# In[42]:


employee_retention_Df['WorkConditionsPostCOVID'].unique()


# In[43]:


#categories imply an inherent order related to improvement or deterioration; therefore, ordinal encoding is used

#mapping dictionary
conditions_mapping = {
    'Verbessert': 3,
    'Weder noch': 2,
    'Verschlimmert': 1
}

#apply mapping to the column
employee_retention_Df['WorkConditionsPostCOVID_Numeric'] = employee_retention_Df['WorkConditionsPostCOVID'].map(conditions_mapping)
employee_retention_Df


# In[44]:


employee_retention_Df['ImprovementMeasures'].unique()


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[46]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline


# In[47]:


import nltk
nltk.download('stopwords')


# In[48]:


from nltk.corpus import stopwords


# In[49]:


#get German stop words from nltk
german_stop_words = stopwords.words('german')


# In[50]:


#measures analysis

data = ['Coaching psychische Stresssituationen',
       'Schutzmaßnahmen wurden erhöht, längere Pausen',
       'Unterstützung der psychischen Gesundheit',
       'Weiterbildungmöglichkeiten erhöht, Schutzmaßnahmen verstärkt', 'Token für Home Office',
       'neue Fort- und Weiterbildungsangebote, zusätzliche Einstellung von Physician Assistants, Delegation von administrativen Aufgaben auf andere Berufsgruppen',
       'Programm zur Unterstützung der psychischen Gesundheit',
       'Mehr Mitarbeiter, da ich der eigene Chef bin, geht es mir gut ',
       'flexible Arbeitszeitmodelle',
       'Regenerationsphasen erhöht, Hygienemassnahmen verbessert\n\n\n',
       'Testmöglichkeiten, längere Regenerationsphasen',
       'Informationsangebot vergrößert, Testpflicht, Hygienemassnahmen erweitert',
       'Psychische Unterstützung, erhöhte Hygienemassnahmen, erhöhte Information',
       'verstärkte Förderung zur Weiterbildung, psychiatrische Hilfe, verstärkte Hygienemaßnahmen',
       'Fortbildung und Informationsbereitschaft vergrößert, intensivere Hygienemassnahmen durchgesetzt',
       'wöchentliche Informationen über RKI-Portal, erhöhte Infektionsschutzmaßnahmen, Ruhepausen nach Einsätzen bei Covid-Patienten',
       'Sportangebote durch die Klinik',
       'erhöhte Hygiene- und Schutzmaßnahmen, monetäre Entschädigungen als Risikozulage',
       'Mitarbeiterinformation zum Covid-Status, verbesserte Hygienemaßnahmen',
       'erhöhter Informationsaustausch im Klinikum, psychische Betreuung und Beratung, erhöhter Gesundheitsschutz',
       'erhöhte Informations - und Hygienemassnahmen',
       'Schutzmaßnahmen',
       'Viele Konferenzen und Besprechungen jetzt online, Fortbildungsveranstaltungen hybrid']

df = pd.DataFrame(data, columns=["Measures_taken"])


# In[51]:


vectorizer = TfidfVectorizer(stop_words=german_stop_words)
kmeans = KMeans(n_clusters=3, random_state=42)
svd = TruncatedSVD(n_components=2)  

pipeline = make_pipeline(vectorizer, svd, kmeans)

df["Measures_taken"].fillna("", inplace=True)

df['Measures_Cluster'] = pipeline.fit_predict(df["Measures_taken"])

print(df[["Measures_taken", 'Measures_Cluster']])


# In[52]:


#Comprehensive Enhancements
cluster_0_data = df[df['Measures_Cluster'] == 0]
cluster_0_data


# In[53]:


#Enhanced Safety and Training Initiatives
cluster_1_data = df[df['Measures_Cluster'] == 1]
cluster_1_data


# In[54]:


#Mental Health Support
cluster_2_data = df[df['Measures_Cluster'] == 2]
cluster_2_data


# In[55]:


cluster0_lst = cluster_0_data['Measures_taken'].unique()
cluster0_lst


# In[56]:


cluster1_lst = cluster_1_data['Measures_taken'].unique()
cluster1_lst


# In[57]:


cluster2_lst = cluster_2_data['Measures_taken'].unique()
cluster2_lst


# In[58]:


employee_retention_Df['Measures_clusters'] = ""
employee_retention_Df


# In[59]:


employee_retention_Df.loc[employee_retention_Df['ImprovementMeasures'].isin(cluster0_lst), "Measures_clusters"] = 0
employee_retention_Df


# In[60]:


employee_retention_Df.loc[employee_retention_Df['ImprovementMeasures'].isin(cluster1_lst), "Measures_clusters"] = 1
employee_retention_Df


# In[61]:


employee_retention_Df.loc[employee_retention_Df['ImprovementMeasures'].isin(cluster2_lst), "Measures_clusters"] = 2
employee_retention_Df


# In[62]:


employee_retention_Df["Measures_clusters"]


# In[63]:


measures_counts = employee_retention_Df["Measures_clusters"].value_counts()
measures_counts


# In[64]:


filtered_rows = employee_retention_Df[employee_retention_Df['ImprovementMeasures'].notnull() & (employee_retention_Df["Measures_clusters"] == "")]
filtered_rows


# In[65]:


filtered_rows[['ImprovementMeasures', "Measures_clusters"]]


# In[66]:


employee_retention_Df.columns


# In[67]:


#calculating the affective retention component for each row
employee_retention_Df["Affective_employee_retention"] = (employee_retention_Df["EnjoyWorkingHere"]+employee_retention_Df["PersonalConnectionWithEmployer"]+employee_retention_Df["RegretIfEmploymentEnded"]+employee_retention_Df["IdentifyWithEmployer"]+employee_retention_Df["ImportanceOfWorkRelationships"])/5
employee_retention_Df


# In[68]:


#calculating the cognitive retention component for each row
employee_retention_Df["Cognitive_employee_retention"] = (employee_retention_Df["SwitchingCostsTimeInvestment"]+employee_retention_Df["DependenceOnEmployerForAlternatives"]+employee_retention_Df["LossOfInvestmentUponSwitching"]+employee_retention_Df["SwitchingCostsPerceived"])/4
employee_retention_Df


# In[69]:


#calculating the normative retention component for each row
employee_retention_Df["Normative_employee_retention"] = (employee_retention_Df["FairnessInStayingWithEmployer"]+employee_retention_Df["ObligationDueToLongRelationship"]+employee_retention_Df["FeelingOfFairnessToEmployer"]+employee_retention_Df["MoralObligationsToEmployer"])/4
employee_retention_Df


# In[70]:


#calculating the contractual retention component for each row
employee_retention_Df["Contractual_employee_retention"] = (employee_retention_Df["ContractualBinding"]+employee_retention_Df["FairWorkContractConditions"]+employee_retention_Df["FairMonetaryCompensation"]+employee_retention_Df["DesireToModifyWorkContract"])/4
employee_retention_Df


# In[71]:


print(employee_retention_Df.columns)


# In[72]:


pip install pingouin


# In[73]:


import pingouin as pg


# In[74]:


#create subset for analyses
columns_to_include = [
    'Age','EnjoyWorkingHere',
    'PersonalConnectionWithEmployer', 'RegretIfEmploymentEnded',
    'IdentifyWithEmployer', 'ImportanceOfWorkRelationships',
    'SwitchingCostsTimeInvestment', 'DependenceOnEmployerForAlternatives',
    'LossOfInvestmentUponSwitching', 'SwitchingCostsPerceived',
    'FairnessInStayingWithEmployer', 'ObligationDueToLongRelationship',
    'FeelingOfFairnessToEmployer', 'MoralObligationsToEmployer',
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation', 'DesireToModifyWorkContract',
    'Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_Church Hospital',
    'CurrentEmployerType_Municipal Hospital',
    'CurrentEmployerType_Other Inpatient Facility',
    'CurrentEmployerType_Outpatient Facility',
    'CurrentEmployerType_Private Hospital',
    'CurrentEmployerType_University Hospital',
    'FacilityLocationType_encoded', 'CurrentPosition_encoded',
    'CurrentWorkModel_encoded', 'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_Mostly Paid',
    'OvertimeCompensation_Mostly Time Off', 'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded', 'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric', 
    'Affective_employee_retention', 'Cognitive_employee_retention',
    'Normative_employee_retention', 'Contractual_employee_retention'
]

subset_employee_retention_Df = employee_retention_Df[columns_to_include]
subset_employee_retention_Df


# In[75]:


print(subset_employee_retention_Df.dtypes)


# In[76]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_employee_retention_Df, ci=.99)


# In[77]:


#create subset for Cronbach Alpha
columns_to_include = [
    'EnjoyWorkingHere',
    'PersonalConnectionWithEmployer', 'RegretIfEmploymentEnded',
    'IdentifyWithEmployer', 'ImportanceOfWorkRelationships',
    'SwitchingCostsTimeInvestment', 'DependenceOnEmployerForAlternatives',
    'LossOfInvestmentUponSwitching', 'SwitchingCostsPerceived',
    'FairnessInStayingWithEmployer', 'ObligationDueToLongRelationship',
    'FeelingOfFairnessToEmployer', 'MoralObligationsToEmployer',
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation', 'DesireToModifyWorkContract'
]

subset_cronbach_alpha_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_Df


# In[78]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_Df, ci=.99)


# In[79]:


#create subset for Cronbach Alpha (Affective Employee Retention)
columns_to_include = [
    'EnjoyWorkingHere',
    'PersonalConnectionWithEmployer', 'RegretIfEmploymentEnded',
    'IdentifyWithEmployer', 'ImportanceOfWorkRelationships'
]

subset_cronbach_alpha_AER_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_AER_Df


# In[80]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_AER_Df, ci=.99)


# In[81]:


#create subset for Cronbach Alpha (Cognitive Employee Retention)
columns_to_include = [
    'SwitchingCostsTimeInvestment', 'DependenceOnEmployerForAlternatives',
    'LossOfInvestmentUponSwitching', 'SwitchingCostsPerceived'
]

subset_cronbach_alpha_CER_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_CER_Df


# In[82]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_CER_Df, ci=.99)


# In[83]:


#create subset for Cronbach Alpha (Normative Employee Retention)
columns_to_include = [
    'FairnessInStayingWithEmployer', 'ObligationDueToLongRelationship',
    'FeelingOfFairnessToEmployer', 'MoralObligationsToEmployer'
]

subset_cronbach_alpha_NER_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_NER_Df


# In[84]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_NER_Df, ci=.99)


# In[85]:


#create subset for Cronbach Alpha (Contractual Employee Retention)
columns_to_include = [
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation', 'DesireToModifyWorkContract'
]

subset_cronbach_alpha_COER_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_COER_Df


# In[86]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_COER_Df, ci=.99)


# In[87]:


#create subset for Cronbach Alpha (Contractual Employee Retention)
columns_to_include = [
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation'
]

subset_cronbach_alpha_COER2_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha_COER2_Df


# In[88]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha_COER2_Df, ci=.99)


# In[89]:


#create subset for Cronbach Alpha
columns_to_include = [
    'Affective_employee_retention', 'Cognitive_employee_retention',
    'Normative_employee_retention', 'Contractual_employee_retention'
]

subset_cronbach_alpha2_Df = employee_retention_Df[columns_to_include]
subset_cronbach_alpha2_Df


# In[90]:


#calculate Cronbach's Alpha and corresponding 99% confidence interval
pg.cronbach_alpha(data=subset_cronbach_alpha2_Df, ci=.99)


# In[91]:


subset_corr = subset_cronbach_alpha2_Df.corr()
subset_corr


# In[92]:


import seaborn as sns
mask = np.triu(np.ones_like(subset_corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(subset_corr, mask=mask, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# In[93]:


corr_matrix = subset_cronbach_alpha_Df.corr()
corr_matrix


# In[168]:


mask2 = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix, mask=mask2, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"size": 8})

plt.title('Correlation Heatmap of Variables')
plt.show()


# In[95]:


#create subset without item 17
columns_to_include = [
    'EnjoyWorkingHere',
    'PersonalConnectionWithEmployer', 'RegretIfEmploymentEnded',
    'IdentifyWithEmployer', 'ImportanceOfWorkRelationships',
    'SwitchingCostsTimeInvestment', 'DependenceOnEmployerForAlternatives',
    'LossOfInvestmentUponSwitching', 'SwitchingCostsPerceived',
    'FairnessInStayingWithEmployer', 'ObligationDueToLongRelationship',
    'FeelingOfFairnessToEmployer', 'MoralObligationsToEmployer',
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation'
]

subset_cronbach_16items_Df = employee_retention_Df[columns_to_include]
subset_cronbach_16items_Df


# In[96]:


subset_corr_16items = subset_cronbach_16items_Df.corr()
subset_corr_16items


# In[97]:


#calculating the contractual retention component for each row without item 17
employee_retention_Df["Contractual_employee_retention2"] = (employee_retention_Df["ContractualBinding"]+employee_retention_Df["FairWorkContractConditions"]+employee_retention_Df["FairMonetaryCompensation"])/3
employee_retention_Df


# In[98]:


#create subset for second analysis
columns_to_include = [
    'Affective_employee_retention', 'Cognitive_employee_retention',
    'Normative_employee_retention', 'Contractual_employee_retention2'
]

subset_4components_Df = employee_retention_Df[columns_to_include]
subset_4components_Df


# In[99]:


subset_4components_Df_corr = subset_4components_Df.corr()
subset_4components_Df_corr


# In[100]:


mask = np.triu(np.ones_like(subset_4components_Df_corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(subset_4components_Df_corr, mask=mask, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# In[101]:


pip install semopy


# In[102]:


from semopy import Model
from semopy.optimizer import Optimizer


# In[103]:


#define the model specification for CFA
model_spec = """
Affective =~ EnjoyWorkingHere + PersonalConnectionWithEmployer + RegretIfEmploymentEnded + IdentifyWithEmployer + ImportanceOfWorkRelationships
Cognitive =~ SwitchingCostsTimeInvestment + DependenceOnEmployerForAlternatives + LossOfInvestmentUponSwitching + SwitchingCostsPerceived
Normative =~ FairnessInStayingWithEmployer + ObligationDueToLongRelationship + FeelingOfFairnessToEmployer + MoralObligationsToEmployer
Contractual =~ ContractualBinding + FairWorkContractConditions + FairMonetaryCompensation + DesireToModifyWorkContract
"""

#initialize and fit the model
model = Model(model_spec)
model.load_dataset(subset_cronbach_alpha_Df)  

#fit the model to your data
results = model.fit()
results


# In[104]:


#parameter estimates
estimates = model.inspect()
print(estimates)


# In[105]:


#estimate factor scores
factor_scores = model.predict(subset_cronbach_alpha_Df)

print(factor_scores)


# In[106]:


#filter the results to only include factor loadings (relationships between latent variables and observed variables)
factor_loadings = estimates[estimates['op'] == '~']
print(factor_loadings)


# In[107]:


from semopy.stats import calc_stats


# In[108]:


stats = calc_stats(model)
stats


# In[109]:


#second CFA
model_spec2 = """
Affective =~ EnjoyWorkingHere + PersonalConnectionWithEmployer + RegretIfEmploymentEnded + IdentifyWithEmployer + ImportanceOfWorkRelationships
Cognitive =~ SwitchingCostsTimeInvestment + DependenceOnEmployerForAlternatives + LossOfInvestmentUponSwitching + SwitchingCostsPerceived
Normative =~ FairnessInStayingWithEmployer + ObligationDueToLongRelationship + FeelingOfFairnessToEmployer + MoralObligationsToEmployer
Contractual =~ ContractualBinding + FairWorkContractConditions + FairMonetaryCompensation 
"""

#initialize and fit the model
model2 = Model(model_spec2)
model.load_dataset(subset_cronbach_alpha_Df)  

#fit the model to your data
results2 = model2.fit(subset_cronbach_alpha_Df)
results2


# In[110]:


#parameter estimates
estimates2 = model2.inspect()
print(estimates2)


# In[111]:


#filter the results to only include factor loadings (relationships between latent variables and observed variables)
factor_loadings2 = estimates2[estimates2['op'] == '~']
print(factor_loadings2)


# In[112]:


stats2 = calc_stats(model2)
stats2


# In[113]:


#third CFA
model_spec3 = """
Affective =~ EnjoyWorkingHere + PersonalConnectionWithEmployer + RegretIfEmploymentEnded + IdentifyWithEmployer + ImportanceOfWorkRelationships
Cognitive =~ SwitchingCostsTimeInvestment + DependenceOnEmployerForAlternatives + LossOfInvestmentUponSwitching + SwitchingCostsPerceived
Normative =~ FairnessInStayingWithEmployer + ObligationDueToLongRelationship + FeelingOfFairnessToEmployer + MoralObligationsToEmployer
"""

#initialize and fit the model
model3 = Model(model_spec3)
model.load_dataset(subset_cronbach_alpha_Df)  

#fit the model to your data
results3 = model3.fit(subset_cronbach_alpha_Df)
results3


# In[114]:


#parameter estimates
estimates3 = model3.inspect()
print(estimates3)


# In[115]:


#filter the results to only include factor loadings (relationships between latent variables and observed variables)
factor_loadings3 = estimates3[estimates3['op'] == '~']
print(factor_loadings3)


# In[116]:


stats3 = calc_stats(model3)
stats3


# In[117]:


#create subset for SPSS analyses
columns_to_include = [
    'Age','EnjoyWorkingHere',
    'PersonalConnectionWithEmployer', 'RegretIfEmploymentEnded',
    'IdentifyWithEmployer', 'ImportanceOfWorkRelationships',
    'SwitchingCostsTimeInvestment', 'DependenceOnEmployerForAlternatives',
    'LossOfInvestmentUponSwitching', 'SwitchingCostsPerceived',
    'FairnessInStayingWithEmployer', 'ObligationDueToLongRelationship',
    'FeelingOfFairnessToEmployer', 'MoralObligationsToEmployer',
    'ContractualBinding', 'FairWorkContractConditions',
    'FairMonetaryCompensation', 'DesireToModifyWorkContract',
    'Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_Church Hospital',
    'CurrentEmployerType_Municipal Hospital',
    'CurrentEmployerType_Other Inpatient Facility',
    'CurrentEmployerType_Outpatient Facility',
    'CurrentEmployerType_Private Hospital',
    'CurrentEmployerType_University Hospital',
    'FacilityLocationType_encoded', 'CurrentPosition_encoded',
    'CurrentWorkModel_encoded', 'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_Mostly Paid',
    'OvertimeCompensation_Mostly Time Off', 'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded', 'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric'
]

subset_employee_retention = employee_retention_Df[columns_to_include]
subset_employee_retention


# In[118]:


subset_employee_retention.to_csv("C:/CarolineZiegler/Studium_DCU/8. Semester/Business Project/Primary Data/subsetSPSS.csv", index = False)


# In[119]:


inpPath2 = "C:/CarolineZiegler/Studium_DCU/8. Semester/Business Project/Primary Data/"
ER_analysis_Df = pd.read_csv(inpPath + "EmployeeRetentionData_C.csv", delimiter =  ",", header = 0, encoding="latin1")
ER_analysis_Df


# In[120]:


for column_name in ER_analysis_Df.columns:
    print(column_name)


# In[121]:


#column cleaning
ER_analysis_Df.drop(["CaseNo"], inplace=True, axis=1)
ER_analysis_Df


# In[122]:


ER_analysis_Df.rename(columns={"ï»¿Age": "Age"}, inplace=True)
ER_analysis_Df


# In[123]:


#descriptive Analysis of all variables
round(ER_analysis_Df.describe(),2)


# In[124]:


#create subset for linear regression and MANOVA analyses
columns_to_include = [
    'Age','Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric',
    'Contractual_Retention',
    'Normative_Retention',
    'Cognitive_Retention',
    'Affective_Retention'
]

subset_ER_analysis_Df = ER_analysis_Df[columns_to_include]
subset_ER_analysis_Df


# In[160]:


#correlation heatmap


# In[162]:


subset_ER_analysis_Df_corr = subset_ER_analysis_Df.corr()
subset_ER_analysis_Df_corr


# In[167]:


mask = np.triu(np.ones_like(subset_ER_analysis_Df_corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(subset_ER_analysis_Df_corr, mask=mask, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, annot_kws={"size": 6})

plt.title('Correlation Heatmap')
plt.show()


# In[125]:


#linear regressions to analyse impact of demographic variables (rentention influential factors) on each employee retention

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[126]:


#affective retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Affective_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_affective = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_affective.score(X_test, y_test))
print(reg_lin_affective.coef_)


# In[127]:


#cognitive retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Cognitive_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_cognitive = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_cognitive.score(X_test, y_test))
print(reg_lin_cognitive.coef_)


# In[128]:


#normative retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Normative_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_normative = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_normative.score(X_test, y_test))
print(reg_lin_normative.coef_)


# In[129]:


#contractual retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded', 'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Contractual_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_contractual = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_contractual.score(X_test, y_test))
print(reg_lin_contractual.coef_)


# In[130]:


#testing demographics


# In[131]:


#affective retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded']]
yDf = subset_ER_analysis_Df['Affective_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_affective_demo = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_affective_demo.score(X_test, y_test))
print(reg_lin_affective_demo.coef_)


# In[132]:


#cognitive retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded']]
yDf = subset_ER_analysis_Df['Cognitive_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_cognitive_demo = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_cognitive_demo.score(X_test, y_test))
print(reg_lin_cognitive_demo.coef_)


# In[133]:


#normative retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded']]
yDf = subset_ER_analysis_Df['Normative_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_normative_demo = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_normative_demo.score(X_test, y_test))
print(reg_lin_normative_demo.coef_)


# In[134]:


#contractual retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['Age','Gender_encoded']]
yDf = subset_ER_analysis_Df['Contractual_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_contractual_demo = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_contractual_demo.score(X_test, y_test))
print(reg_lin_contractual_demo.coef_)


# In[135]:


#financial factor analysis


# In[136]:


#affective retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither']]
yDf = subset_ER_analysis_Df['Affective_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_affective_fin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_affective_fin.score(X_test, y_test))
print(reg_lin_affective_fin.coef_)


# In[137]:


#cognitive retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither']]
yDf = subset_ER_analysis_Df['Cognitive_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_cognitive_fin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_cognitive_fin.score(X_test, y_test))
print(reg_lin_cognitive_fin.coef_)


# In[138]:


#normative retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither']]
yDf = subset_ER_analysis_Df['Normative_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_normative_fin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_normative_fin.score(X_test, y_test))
print(reg_lin_normative_fin.coef_)


# In[139]:


#contractual retention analysis
#split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = subset_ER_analysis_Df[['OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither']]
yDf = subset_ER_analysis_Df['Contractual_Retention']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_contractual_fin = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_contractual_fin.score(X_test, y_test))
print(reg_lin_contractual_fin.coef_)


# In[140]:


pip install statsmodels pandas


# In[141]:


import statsmodels.api as sm


# In[145]:


#linear regression for affective retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Affective_Retention']

#fit the linear regression model
model_affective = sm.OLS(yDf, XDf).fit()

print(model_affective.summary())


# In[146]:


#linear regression for normative retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Normative_Retention']

#fit the linear regression model
model_normative = sm.OLS(yDf, XDf).fit()

print(model_normative.summary())


# In[147]:


#linear regression for normative retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Cognitive_Retention']

#fit the linear regression model
model_cognitive = sm.OLS(yDf, XDf).fit()

print(model_cognitive.summary())


# In[148]:


#linear regression for contractual retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Contractual_Retention']

#fit the linear regression model
model_contractual = sm.OLS(yDf, XDf).fit()

print(model_contractual.summary())


# In[149]:


#testing demographics


# In[150]:


#linear regression for affective retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Affective_Retention']

#fit the linear regression model
model_affective_demo = sm.OLS(yDf, XDf).fit()

print(model_affective_demo.summary())


# In[151]:


#linear regression for normative retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Normative_Retention']

#fit the linear regression model
model_normative_demo = sm.OLS(yDf, XDf).fit()

print(model_normative_demo.summary())


# In[152]:


#linear regression for cognitive retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Cognitive_Retention']

#fit the linear regression model
model_cognitive_demo = sm.OLS(yDf, XDf).fit()

print(model_cognitive_demo.summary())


# In[153]:


#linear regression for contractual retention
#define the independent variables
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded']]

#add constant to the model (the intercept)
XDf = sm.add_constant(XDf)

#define the dependent variable
yDf = subset_ER_analysis_Df['Contractual_Retention']

#fit the linear regression model
model_contractual_demo = sm.OLS(yDf, XDf).fit()

print(model_contractual_demo.summary())


# In[154]:


#stepwise regression backward elimination to identify for each component the most influencial factors


# In[155]:


#affective retention
#define your independent variables (XDf) and dependent variable (yDf)
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Affective_Retention']

#add constant to XDf to include an intercept in the model
XDf = sm.add_constant(XDf)

#set significance level
sl = 0.05

#perform backward elimination
columns = list(XDf.columns)
while len(columns) > 0:
    #fit the model
    X_opt = XDf[columns]
    model = sm.OLS(yDf, X_opt).fit()
    #get the p-values for the fitted model and find the max p-value
    p_values = model.pvalues
    max_p_value = max(p_values)
    feature_with_max_p_value = p_values.idxmax()
    #if the max p-value is greater than the significance level, remove the feature
    if max_p_value > sl:
        columns.remove(feature_with_max_p_value)
    else:
        break

#fit the model with the selected features
X_opt = XDf[columns]
model_affective_srbe = sm.OLS(yDf, X_opt).fit()

#display the final model summary
print(model_affective_srbe.summary())


# In[156]:


#affective retention
#define your independent variables (XDf) and dependent variable (yDf)
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Affective_Retention']

#add constant to XDf to include an intercept in the model
XDf = sm.add_constant(XDf)

#set significance level
sl = 0.1

#perform backward elimination
columns = list(XDf.columns)
while len(columns) > 0:
    #fit the model
    X_opt = XDf[columns]
    model = sm.OLS(yDf, X_opt).fit()
    #get the p-values for the fitted model and find the max p-value
    p_values = model.pvalues
    max_p_value = max(p_values)
    feature_with_max_p_value = p_values.idxmax()
    #if the max p-value is greater than the significance level, remove the feature
    if max_p_value > sl:
        columns.remove(feature_with_max_p_value)
    else:
        break

#fit the model with the selected features
X_opt = XDf[columns]
model_affective_srbe2 = sm.OLS(yDf, X_opt).fit()

#display the final model summary
print(model_affective_srbe2.summary())


# In[157]:


#normative retention
#define your independent variables (XDf) and dependent variable (yDf)
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Normative_Retention']

#add constant to XDf to include an intercept in the model
XDf = sm.add_constant(XDf)

#set significance level
sl = 0.05

#perform backward elimination
columns = list(XDf.columns)
while len(columns) > 0:
    #fit the model
    X_opt = XDf[columns]
    model = sm.OLS(yDf, X_opt).fit()
    #get the p-values for the fitted model and find the max p-value
    p_values = model.pvalues
    max_p_value = max(p_values)
    feature_with_max_p_value = p_values.idxmax()
    #if the max p-value is greater than the significance level, remove the feature
    if max_p_value > sl:
        columns.remove(feature_with_max_p_value)
    else:
        break

#fit the model with the selected features
X_opt = XDf[columns]
model_normative_srbe = sm.OLS(yDf, X_opt).fit()

#display the final model summary
print(model_normative_srbe.summary())


# In[158]:


#cognitive retention
#define your independent variables (XDf) and dependent variable (yDf)
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Cognitive_Retention']

#add constant to XDf to include an intercept in the model
XDf = sm.add_constant(XDf)

#set significance level
sl = 0.05

#perform backward elimination
columns = list(XDf.columns)
while len(columns) > 0:
    #fit the model
    X_opt = XDf[columns]
    model = sm.OLS(yDf, X_opt).fit()
    #get the p-values for the fitted model and find the max p-value
    p_values = model.pvalues
    max_p_value = max(p_values)
    feature_with_max_p_value = p_values.idxmax()
    #if the max p-value is greater than the significance level, remove the feature
    if max_p_value > sl:
        columns.remove(feature_with_max_p_value)
    else:
        break

#fit the model with the selected features
X_opt = XDf[columns]
model_cognitive_srbe = sm.OLS(yDf, X_opt).fit()

#display the final model summary
print(model_cognitive_srbe.summary())


# In[159]:


#contractual retention
#define your independent variables (XDf) and dependent variable (yDf)
XDf = subset_ER_analysis_Df[['Age', 'Gender_encoded', 'Experience_encoded',
        'CurrentEmployerType_ChurchHospital',
        'CurrentEmployerType_MunicipalHospital',
        'CurrentEmployerType_OtherInpatientFacility',
        'CurrentEmployerType_OutpatientFacility',
        'CurrentEmployerType_PrivateHospital',
        'CurrentEmployerType_UniversityHospital',
        'FacilityLocationType_encoded',
        'CurrentPosition_encoded',
        'CurrentWorkModel_encoded',
        'AverageWeeklyWorkingHours_encoded',
        'OvertimeCompensation_MostlyPaid',
        'OvertimeCompensation_MostlyTimeOff',
        'OvertimeCompensation_Neither',
        'CurrentWorkConditionsRating_encoded',
        'StaffingLevelRating_encoded',
        'WorkConditionsPostCOVID_Numeric']]
yDf = subset_ER_analysis_Df['Contractual_Retention']

#add constant to XDf to include an intercept in the model
XDf = sm.add_constant(XDf)

#set significance level
sl = 0.05

#perform backward elimination
columns = list(XDf.columns)
while len(columns) > 0:
    #fit the model
    X_opt = XDf[columns]
    model = sm.OLS(yDf, X_opt).fit()
    #get the p-values for the fitted model and find the max p-value
    p_values = model.pvalues
    max_p_value = max(p_values)
    feature_with_max_p_value = p_values.idxmax()
    #if the max p-value is greater than the significance level, remove the feature
    if max_p_value > sl:
        columns.remove(feature_with_max_p_value)
    else:
        break

#fit the model with the selected features
X_opt = XDf[columns]
model_contractual_srbe = sm.OLS(yDf, X_opt).fit()

#display the final model summary
print(model_contractual_srbe.summary())


# In[169]:


#MANCOVA for a holistic analysis of factor influence
from statsmodels.multivariate.manova import MANOVA


# In[171]:


#define independent variables
independent_vars = [
    'Experience_encoded',
    'CurrentEmployerType_ChurchHospital',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OtherInpatientFacility',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_PrivateHospital',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyPaid',
    'OvertimeCompensation_MostlyTimeOff',
    'OvertimeCompensation_Neither',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric'
]

#covariates
covariates = ['Age', 'Gender_encoded']

#dependent variables
dependent_vars = [
    'Contractual_Retention',
    'Normative_Retention',
    'Cognitive_Retention',
    'Affective_Retention'
]

#combine all predictors including covariates for the model equation
all_predictors = independent_vars + covariates
predictors_str = ' + '.join(all_predictors)

#define the model equation for MANCOVA
#dependent variables are combined using '+', and '~' separates them from the predictors
model_eq = f"{' + '.join(dependent_vars)} ~ {predictors_str}"

#creating and fitting the MANCOVA model
mv = MANOVA.from_formula(model_eq, data=subset_ER_analysis_Df)
result = mv.mv_test()

# Printing the results
print(result)


# In[174]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[175]:


#filter the DataFrame to include only your independent variables
X = subset_ER_analysis_Df[independent_vars]

#adding a constant is necessary for the intercept term when calculating VIF
X = add_constant(X)

#calculating VIF for each independent variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Displaying the VIF for each variable
print(vif)


# In[176]:


#define independent variables
independent_vars = [
    'Experience_encoded',
    'CurrentEmployerType_MunicipalHospital',
    'CurrentEmployerType_OutpatientFacility',
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyTimeOff',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric'
]

#covariates
covariates = ['Age', 'Gender_encoded']

#dependent variables
dependent_vars = [
    'Contractual_Retention',
    'Normative_Retention',
    'Cognitive_Retention',
    'Affective_Retention'
]

#combine all predictors including covariates for the model equation
all_predictors = independent_vars + covariates
predictors_str = ' + '.join(all_predictors)

#define the model equation for MANCOVA
#dependent variables are combined using '+', and '~' separates them from the predictors
model_eq = f"{' + '.join(dependent_vars)} ~ {predictors_str}"

#creating and fitting the MANCOVA model
mv = MANOVA.from_formula(model_eq, data=subset_ER_analysis_Df)
result = mv.mv_test()

# Printing the results
print(result)


# In[177]:


#define independent variables
independent_vars = [
    'CurrentEmployerType_UniversityHospital',
    'FacilityLocationType_encoded',
    'CurrentPosition_encoded',
    'CurrentWorkModel_encoded',
    'AverageWeeklyWorkingHours_encoded',
    'OvertimeCompensation_MostlyTimeOff',
    'CurrentWorkConditionsRating_encoded',
    'StaffingLevelRating_encoded',
    'WorkConditionsPostCOVID_Numeric'
]

#covariates
covariates = ['Age', 'Gender_encoded']

#dependent variables
dependent_vars = [
    'Contractual_Retention',
    'Normative_Retention',
    'Cognitive_Retention',
    'Affective_Retention'
]

#combine all predictors including covariates for the model equation
all_predictors = independent_vars + covariates
predictors_str = ' + '.join(all_predictors)

#define the model equation for MANCOVA
#dependent variables are combined using '+', and '~' separates them from the predictors
model_eq = f"{' + '.join(dependent_vars)} ~ {predictors_str}"

#creating and fitting the MANCOVA model
mv = MANOVA.from_formula(model_eq, data=subset_ER_analysis_Df)
result = mv.mv_test()

# Printing the results
print(result)


# In[178]:


subset_ER_analysis_Df


# In[179]:


subset_ER_analysis_Df["Age"].unique()


# In[181]:


conditions = [
    (subset_ER_analysis_Df["Age"] <= 27),
    (subset_ER_analysis_Df["Age"] >= 28) & (subset_ER_analysis_Df["Age"] <= 43),
    (subset_ER_analysis_Df["Age"] >= 44) & (subset_ER_analysis_Df["Age"] <= 59),
    (subset_ER_analysis_Df["Age"] >= 60) & (subset_ER_analysis_Df["Age"] <= 79)
]

values = ["20-27", "28-43", "44-59", "60-79"]

subset_ER_analysis_Df["Generation"] = np.select(conditions, values, default="Unknown")
subset_ER_analysis_Df


# In[182]:


subset_ER_analysis_Df["Generation"].unique()


# In[183]:


#calculating the mean for each generation group for the specified columns
mean_values = subset_ER_analysis_Df.groupby('Generation')[['Contractual_Retention', 'Normative_Retention', 'Cognitive_Retention', 'Affective_Retention']].mean()

#display the mean values
print(mean_values)


# In[ ]:




