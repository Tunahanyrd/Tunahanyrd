# software engineering & data systems portfolio

a collection of projects, notes, and experiments from my shift toward backend engineering, postgresql, and data-intensive systems.

this repository used to be mostly a machine learning / deep learning archive. that was useful while i was exploring ai, graphs, computer vision, and kaggle-style workflows, but my current focus has changed.

these days, i am mostly interested in:

* backend systems
* postgresql and relational modeling
* transaction safety
* database-backed application design
* data-intensive systems
* storage engines and database internals
* technical writing around engineering trade-offs

## current direction

i am a computer science student focused on backend and data-intensive systems. my recent work is less about training fancy models and more about building systems that keep data consistent, model real workflows properly, and survive edge cases.

the main things i care about right now:

* designing schemas that actually encode business rules
* using postgresql constraints, indexes, triggers, and transactions properly
* understanding how oltp systems behave under real usage
* writing backend services with clear boundaries
* learning database internals instead of treating databases like dumb storage

## active / recent projects

### obs-go

student information system mvp built with go and postgresql.

this project is one of my main backend/database projects. the goal was not just to build crud endpoints, but to model academic workflows properly and push consistency rules into the database where they belong.

highlights:

* layered go backend with rest handlers, service logic, and repository layer
* postgresql-backed data model for student information system workflows
* transaction-safe course enrollment
* use of `pg_advisory_xact_lock` to prevent race conditions during concurrent registration
* schema constraints, triggers, audit logs, materialized views, and exclusion constraints
* focus on consistency, transaction boundaries, and database-first design

[repo](github.com/tunahanyrd/obs-go)

### koru

a current software project i am actively building.

this is part of my newer direction: practical backend engineering, system design, and building real applications instead of only running experiments in notebooks.

[repo](github.com/tunahanyrd/koru)

### neuropass

1st place project at neurobridge hackathon.

an ml-based drug screening pipeline built from smiles inputs. the project included dataset cleaning, feature preparation, model training, and prediction apis for bbb permeability, logbb exposure, and tox21 toxicity risk.

even though i am not mainly focused on ai anymore, this project still matters because it combined applied ml with backend/api work and interpretable outputs.

highlights:

* python, fastapi, scikit-learn, docker
* trained and compared 18 classical ml models
* used extratrees-based models for internal evaluation
* combined probabilistic model outputs with deterministic medicinal-chemistry rules
* produced interpretable screening results and risk explanations

## technical writing

i also write about computer science and engineering trade-offs at:

funeralcs.com

topics i have written or explored include:

* lsm-tree vs. b-tree storage engines
* llm hallucinations
* oltp workloads
* indexing trade-offs
* write-ahead logging
* bloom filters
* compaction
* database internals

writing helps me turn “i kind of understand this” into “i can explain the trade-off clearly.”

## older learning archive

this repository also contains older machine learning and deep learning projects.

i am keeping them because they show my learning path, not because they represent my current direction.

### graph neural networks

older experiments with graph-based learning, including relational gat, fake news detection, recommendation systems, and graph visualizations.

these projects taught me that graph methods can generalize well, but they also come with serious hardware and implementation costs.

### computer vision

older experiments with medical imaging, gradcam, transfer learning, and few-shot learning.

examples:

* brain tumor classification
* car recognition
* covid image analysis

### competitions

kaggle and datathon-style experiments.

these folders are messy by nature. competitions are usually about iteration, feature engineering, model comparison, and squeezing small performance gains out of chaotic workflows.

### data analysis

older tabular and visualization work.

the microsoft malware prediction analysis was especially important for me because it pushed me back toward data, tabular problems, and deeper system-level thinking.

## tech stack

current focus:

* languages: go, python, sql
* databases: postgresql, duckdb
* backend: fastapi, rest apis, sqlalchemy, alembic, docker
* data: pandas, numpy, data modeling, query optimization
* tools: linux, git

previous / learning projects:

* pytorch
* pytorch geometric
* scikit-learn
* catboost
* xgboost
* lightgbm
* sbert
* transformers
* matplotlib
* seaborn

## what changed

i used to describe this as a deep learning and machine learning portfolio.

that is no longer accurate.

ai/ml was an important learning phase for me, but my current direction is backend engineering and data systems. i am more interested in how data is modeled, stored, queried, indexed, protected by transactions, and exposed through clean services.

in short:

i care less about chasing model accuracy now, and more about building reliable systems around data.

## links

github: github.com/tunahanyrd
website: funeralcs.com
linkedin: linkedin.com/in/tunahanyrd
