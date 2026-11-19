# Frontend Test Suites

Frontend unit and integration tests are executed via **Vitest** with `@vitejs/plugin-react`.

Per project convention (`vitest.config.ts`), test specs are co-located alongside their respective UI components inside `src/`:

- `src/app/smoke.test.ts`: Top-level application shell smoke test.
- `src/app/unifiedIntegration.test.ts`: End-to-end Studio page integration test.
- `src/features/insights/insights.test.ts`: Observability & telemetry dashboard tests.
- `src/features/compression/client.test.ts`: Lexical compression UI client tests.
- `src/features/routing/helpers.test.ts`: Routing heuristics & tier mapping tests.
- `src/features/routing/client.test.ts`: Model routing API client tests.

### Running Frontend Tests
```bash
npm test
# Or watch mode:
npx vitest
```
