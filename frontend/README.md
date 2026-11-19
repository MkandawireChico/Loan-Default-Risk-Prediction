# Nyanja Credit Risk Intelligence frontend

React, TypeScript and Vite application for the synthetic credit-risk platform.

## Local development

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the FastAPI service at `http://localhost:8000`. Override the API base URL with a Vite environment variable:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

## Workspaces

- Portfolio overview: live synthetic portfolio summary and segment/branch breakdowns.
- Application assessment: editable synthetic sample profile and model inference.
- Batch analysis: CSV scoring through the API with downloadable results.
- Model governance: held-out test metrics, operating point and feature governance.

All data and model outputs are synthetic demonstrations. Do not upload real customer or banking records.
