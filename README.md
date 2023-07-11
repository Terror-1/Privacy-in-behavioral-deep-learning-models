# Differential Privacy in mouse and keyboard dataset
- created by Hossam Elfar
## Introduction 
Differential privacy (DP) is a system designed to publicly share information about a dataset while preserving the privacy of individuals within the dataset. It achieves this by describing patterns of groups within the dataset without revealing specific information about individuals. The fundamental concept behind differential privacy is that even with access to the query results, it should be difficult to infer sensitive information about any individual in the dataset, ensuring their privacy.

Another perspective on differential privacy is that it serves as a constraint on algorithms used to disclose aggregate information from a statistical database, preventing the disclosure of private information related to specific records in the database. Government agencies and companies both use this method to publish demographic and statistical data, while also ensuring the confidentiality of survey responses and user behaviour information. It allows for control over visibility, even for internal analysts.

Essentially, an algorithm is considered differentially private if an observer, based on the output it produces, cannot determine whether a particular individual's information was used in the computation. While differential privacy is frequently discussed in the context of identifying individuals in a database, it also provides provable resistance against identification and reidentification attacks.[3]

Differential privacy originated from the field of cryptography, developed by cryptographers, and thus shares much of its terminology and concepts with cryptography.

