# Tunahan Yardimci

Computer Science student focused on backend engineering and data-intensive systems. I work mostly with PostgreSQL, Go, Python/FastAPI, and database-backed application design.

Current direction: OLTP systems, schema design, transaction safety, storage-engine concepts, and practical software that keeps data consistent under real workflows.

## What I Work On

- Backend services with clear service boundaries
- PostgreSQL schemas, constraints, indexes, triggers, and transactions
- Data-intensive applications and local-first tools
- Linux desktop utilities and small systems projects
- Technical writing about databases, storage, LLM behavior, and engineering trade-offs

## Selected Projects

### [koru](https://github.com/Tunahanyrd/koru)

Desktop-first, local-first document board for personal records such as invoices, contracts, certificates, scans, and notes.

Koru combines a native Gio desktop app, CLI tooling, SQLite FTS5 search, OCR extraction, content-addressed blob storage, soft delete, backup/restore, and local recovery checks. The design goal is simple: keep original bytes recoverable and searchable without accounts or cloud dependency.

### [IU-KIK-APP](https://github.com/Tunahanyrd/IU-KIK-APP)

Institutional work tracking and coordination system for Istanbul University. Built for structured announcements, task assignment, scheduling, approvals, auditability, and push notifications instead of losing operational work in WhatsApp groups.

Backend: FastAPI, PostgreSQL 16, SQLAlchemy, Alembic, APScheduler, JWT auth, append-only audit ledger, notification outbox, Prometheus metrics.

Frontend: React Native, Expo, role-based navigation, push notifications, dashboard exports, and mobile workflows for staff, managers, and student assistants.

### [obs-go](https://github.com/Tunahanyrd/obs-go)

Student Information System MVP built with Go and PostgreSQL. This project is mainly about modeling core academic workflows correctly, even in a small monolith.

Highlights:

- Layered Go backend with REST handlers, services, repositories, and domain logic
- PostgreSQL schemas for students, terms, courses, enrollments, schedules, audit logs, and statistics
- Transaction-safe enrollment flow using constraints and `pg_advisory_xact_lock`
- Database-level consistency through triggers, materialized views, and GiST exclusion constraints

### [belgin-privacy-filter](https://github.com/Tunahanyrd/belgin-privacy-filter)

Turkish-focused LoRA adapter for privacy-sensitive span detection on top of `openai/privacy-filter`.

It targets Turkish complaint and customer-support style text, detecting spans such as private person names, emails, phone numbers, account numbers, and secrets so downstream systems can mask sensitive data before analytics, indexing, logging, or dataset preparation.

### [neuropass](https://github.com/Tunahanyrd/neuropass)

1st place NeuroBridge Hackathon project for BBB permeability and ADMET screening.

Built an ML-based drug screening pipeline from SMILES inputs with dataset cleaning, feature preparation, model training, FastAPI prediction endpoints, and interpretable output for BBB permeability, logBB exposure, and Tox21 toxicity risk.

## Experience

- Software Engineer, part-time at Istanbul University
- Backend and PostgreSQL schema design for a department-wide academic mobile application
- President, Data Club Community at Istanbul University
- Co-founder and technical writer at [funeralcs.com](https://funeralcs.com)

## Tech Stack

- Languages: Go, Python, SQL, Shell
- Databases: PostgreSQL, SQLite, DuckDB
- Backend: FastAPI, REST APIs, SQLAlchemy, Alembic, Docker
- Data: pandas, NumPy, data modeling, query optimization
- Tools: Linux, Git, desktop integration, small CLIs

## Links

- Website: [funeralcs.com](https://funeralcs.com)
- LinkedIn: [linkedin.com/in/tunahanyrd](https://linkedin.com/in/tunahanyrd)
