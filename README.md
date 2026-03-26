# CheXpertVision (WIP)

Chest X-ray analysis system combining computer vision and LLMs.

## Current Progress

* Dataset: CheXpert
* Selected 5 diseases:

  * Atelectasis
  * Cardiomegaly
  * Consolidation
  * Edema
  * Pleural Effusion

## Label Preprocessing

* NaN values → 0 (treated as absence)
* Uncertain labels (-1) → 1 (U-Ones strategy)

Rationale:

* Prioritize sensitivity (medical context)
* Reduce false negatives

## Status

Work in progress – building data pipeline and baseline model.
