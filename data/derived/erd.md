---
title: PackageFlow ERD (Entity Relationship Diagram)
source_file: PackageFlow - ERD.png
doc_type: erd
converted: 2026-10-20
method: Transcribed from the PNG by Claude, cross-checked against the entity table in PackageFlow_Technical_Architecture_and_Data_Model.docx (section 7). Review before indexing.
---

# PackageFlow ERD (Entity Relationship Diagram)

The PackageFlow database has **13 entities (tables)**. Every primary key is a UUID. Every relationship in the diagram is **one to zero-or-many** (one parent row, any number of child rows).

**Hub tables:**
- `person` is referenced by 9 other tables (who requested, received, handled, delivered, uploaded, responded, owns, or acted).
- `package_record` is the center of the package lifecycle. 8 tables hang off it.

"Purpose" lines come from the entity table in the Technical Architecture and Data Model doc. Fields and relationships come from the ERD image.

---

## Entity: person

**Purpose:** Requester, receiver, delivery team member, Facilities Lead, or other authorized person reference.

| Field | Type | Key |
|---|---|---|
| person_id | UUID | PK |
| enterprise_user_id | VARCHAR | |
| display_name | VARCHAR | |
| contact_channel | VARCHAR | |
| active_flag | BOOLEAN | |

**Referenced by:** procurement_order_ref (requester), package_record (requester), custody_event (from_person, to_person, actor), package_status_event (actor), evidence_item (uploaded_by), notification (recipient), acknowledgement (responder), exception_case (owner), assignment (assigned_to, assigned_by), audit_event (actor).

---

## Entity: location

**Purpose:** Normalized building, floor, room, lab, secure area, or delivery point.

| Field | Type | Key |
|---|---|---|
| location_id | UUID | PK |
| building_code | VARCHAR | |
| floor | VARCHAR | |
| room_code | VARCHAR | |
| location_type | VARCHAR | |
| active_flag | BOOLEAN | |

**Referenced by:** package_record (delivered_to, via location_id).

---

## Entity: procurement_order_ref

**Purpose:** Cached or referenced procurement information used for matching and display.

| Field | Type | Key |
|---|---|---|
| procurement_order_id | UUID | PK |
| vendor_id | UUID | |
| catalogue_id | UUID | |
| requester_id | UUID | FK → person |
| package_type | VARCHAR | |
| work_location | VARCHAR | |
| delivery_instructions | TEXT | |
| source_updated_at | DATETIME | |
| source_version | VARCHAR | |

**Relationships:**
- One person (requester) has zero or many procurement orders.
- One procurement order **contains** zero or many package_records.

Note: vendor_id and catalogue_id are UUIDs but are not foreign keys in this diagram. They point to data in the external procurement system.

---

## Entity: package_record

**Purpose:** One physical package or package unit handled by PackageFlow.

| Field | Type | Key |
|---|---|---|
| package_id | UUID | PK |
| procurement_order_id | UUID | FK → procurement_order_ref |
| requester_id | UUID | FK → person |
| location_id | UUID | FK → location |
| package_sequence | INT | |
| package_count | INT | |
| package_type | VARCHAR | |
| current_state | VARCHAR | |
| received_at | DATETIME | |
| closed_at | DATETIME | |

**Relationships:**
- Belongs to one procurement_order_ref (contains).
- Belongs to one person as requester.
- Delivered to one location (delivered_to).
- Has zero or many: custody_event, package_status_event, evidence_item, job_execution, notification, acknowledgement, exception_case, assignment.

`package_sequence` and `package_count` support one order arriving as several packages (for example, package 2 of 3).

---

## Entity: assignment

**Purpose:** Current and historical delivery ownership.

| Field | Type | Key |
|---|---|---|
| assignment_id | UUID | PK |
| package_id | UUID | FK → package_record |
| assigned_to | UUID | FK → person |
| assigned_by | UUID | FK → person |
| status | VARCHAR | |
| assigned_at | DATETIME | |
| accepted_at | DATETIME | |
| ended_at | DATETIME | |
| reason | VARCHAR | |

**Relationships:** One package has zero or many assignments. Each assignment links two people: who it's assigned to and who assigned it.

---

## Entity: custody_event

**Purpose:** Immutable record of receipt, handoff, acceptance, reassignment, and delivery custody changes.

| Field | Type | Key |
|---|---|---|
| custody_event_id | UUID | PK |
| package_id | UUID | FK → package_record |
| from_person_id | UUID | FK → person |
| to_person_id | UUID | FK → person |
| actor_id | UUID | FK → person |
| event_type | VARCHAR | |
| event_at | DATETIME | |
| notes | TEXT | |

**Relationships:** One package has zero or many custody events. Each event references up to three people: from_person, to_person, and the actor who recorded it.

---

## Entity: package_status_event

**Purpose:** Immutable lifecycle transition history.

| Field | Type | Key |
|---|---|---|
| status_event_id | UUID | PK |
| package_id | UUID | FK → package_record |
| from_state | VARCHAR | |
| to_state | VARCHAR | |
| reason_code | VARCHAR | |
| actor_id | UUID | FK → person |
| occurred_at | DATETIME | |
| source | VARCHAR | |

**Relationships:** One package has zero or many status events. Each event has one actor (person).

---

## Entity: evidence_item

**Purpose:** Metadata for a photo, signature, note, or other proof.

| Field | Type | Key |
|---|---|---|
| evidence_id | UUID | PK |
| package_id | UUID | FK → package_record |
| evidence_type | VARCHAR | |
| storage_uri | VARCHAR | |
| upload_status | VARCHAR | |
| captured_at | DATETIME | |
| uploaded_by | UUID | FK → person |
| checksum | VARCHAR | |
| retention_class | VARCHAR | |

**Relationships:** One package has zero or many evidence items. Each item has one uploader (person).

The file itself lives in object storage (`storage_uri`). This table only holds the metadata.

---

## Entity: notification

**Purpose:** Notification intent and delivery outcome.

| Field | Type | Key |
|---|---|---|
| notification_id | UUID | PK |
| package_id | UUID | FK → package_record |
| recipient_id | UUID | FK → person |
| event_type | VARCHAR | |
| channel | VARCHAR | |
| template_id | VARCHAR | |
| status | VARCHAR | |
| attempt_count | INT | |
| sent_at | DATETIME | |
| last_error | VARCHAR | |

**Relationships:** One package has zero or many notifications. Each notification goes to one recipient (person).

---

## Entity: acknowledgement

**Purpose:** Requester response to delivery.

| Field | Type | Key |
|---|---|---|
| acknowledgement_id | UUID | PK |
| package_id | UUID | FK → package_record |
| responded_by | UUID | FK → person |
| response_type | VARCHAR | |
| responded_at | DATETIME | |
| comment | TEXT | |
| dispute_reason | VARCHAR | |

**Relationships:** One package has zero or many acknowledgements. Each one has one responder (person).

---

## Entity: exception_case

**Purpose:** Problem requiring action, ownership, and resolution.

| Field | Type | Key |
|---|---|---|
| exception_id | UUID | PK |
| package_id | UUID | FK → package_record |
| owner_id | UUID | FK → person |
| category | VARCHAR | |
| severity | VARCHAR | |
| status | VARCHAR | |
| opened_at | DATETIME | |
| resolved_at | DATETIME | |
| resolution_code | VARCHAR | |

**Relationships:** One package has zero or many exception cases. Each case has one owner (person).

---

## Entity: job_execution

**Purpose:** Background job state and retry tracking.

| Field | Type | Key |
|---|---|---|
| job_id | UUID | PK |
| package_id | UUID | FK → package_record |
| job_type | VARCHAR | |
| scheduled_at | DATETIME | |
| started_at | DATETIME | |
| completed_at | DATETIME | |
| status | VARCHAR | |
| attempts | INT | |
| error_code | VARCHAR | |

**Relationships:** One package has zero or many job executions. No link to person.

---

## Entity: audit_event

**Purpose:** Security and operational audit trail for sensitive actions.

| Field | Type | Key |
|---|---|---|
| audit_id | UUID | PK |
| entity_type | VARCHAR | |
| entity_id | UUID | |
| action | VARCHAR | |
| actor_id | UUID | FK → person |
| actor_role | VARCHAR | |
| occurred_at | DATETIME | |
| correlation_id | UUID | |
| metadata | JSON | |

**Relationships:** One person (actor) has zero or many audit events. Not linked directly to package_record. `entity_type` + `entity_id` can point at any table (for example entity_type = "package_record").

---

## All relationships

| Parent (one) | Child (zero or many) | Via field | Label in ERD |
|---|---|---|---|
| person | procurement_order_ref | requester_id | requester |
| person | package_record | requester_id | requester |
| person | custody_event | from_person_id | from_person |
| person | custody_event | to_person_id | to_person |
| person | custody_event | actor_id | actor |
| person | package_status_event | actor_id | actor |
| person | evidence_item | uploaded_by | uploaded_by |
| person | notification | recipient_id | recipient |
| person | acknowledgement | responded_by | responder |
| person | exception_case | owner_id | owner |
| person | assignment | assigned_to | assigned_to |
| person | assignment | assigned_by | assigned_by |
| person | audit_event | actor_id | actor |
| location | package_record | location_id | delivered_to |
| procurement_order_ref | package_record | procurement_order_id | contains |
| package_record | assignment | package_id | |
| package_record | custody_event | package_id | |
| package_record | package_status_event | package_id | |
| package_record | evidence_item | package_id | |
| package_record | notification | package_id | |
| package_record | acknowledgement | package_id | |
| package_record | exception_case | package_id | |
| package_record | job_execution | package_id | |

---

## Differences between the ERD and the Technical Architecture doc

- **package_record location:** the ERD has `location_id` (FK → location). The doc's entity table lists `location_snapshot` instead. These mean different things: a live link to the location table vs. a copy of the location saved at the time. Needs a decision.
- Everything else (fields per entity, 13 entities) matches.
