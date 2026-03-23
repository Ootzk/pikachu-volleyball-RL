# CLAUDE.md

이 프로젝트의 개발 규칙과 컨벤션을 정리한 문서.

## 버전 관리 & 브랜치 전략

- **Semantic Versioning (semver)** 준수: `MAJOR.MINOR.PATCH`
- Git tag로 버전 표기 (예: `v0.1.0`)

### 브랜치 구조

| 브랜치 | 용도 | 머지 방식 |
|--------|------|-----------|
| `main` | 안정 릴리스 상태 유지 | `release/*` → main: merge commit |
| `release/{version}` | 릴리스 단위 통합 브랜치 (예: `release/v0.1.0`) | `feat/*`, `fix/*` → release: squash merge |
| `feat/*`, `fix/*` | 기능/버그 단위 작업 브랜치 | PR로 release 브랜치에 머지 |

### 워크플로우

1. `release/{version}` 브랜치 생성
2. `feat/*`, `fix/*` 브랜치에서 작업 → `release/{version}`으로 PR (squash merge)
3. 릴리스 준비 완료 시 `release/{version}` → `main`으로 PR (merge commit)
4. `main`에 머지 후 버전 tag 생성

## 개발 환경

- **Python**: 3.10+
- **패키지 관리**: uv (`pyproject.toml` + `uv.lock`)
- **핵심 패키지**: pettingzoo, stable-baselines3, torch, gymnasium, numpy
- **하드웨어**: AMD Ryzen 7 3700X (8C/16T), NVIDIA RTX 2080 Super (8GB)
  - 저차원 벡터 관측 + MLP 정책이라 GPU보다 CPU(환경 병렬화)가 병목
  - SB3 `SubprocVecEnv`로 8~16개 환경 병렬 실행 가능

## CI/CD (GitHub Actions)

| 대상 | 트리거 | 내용 |
|------|--------|------|
| Python lint/test | PR, push to main | ruff lint, pytest |
| Web lint/build | PR, push to main | eslint, 빌드 확인 |
| 웹 데모 배포 | push to main (Phase 3) | GitHub Pages 자동 배포 |

- 학습은 CI에서 돌리지 않음 (GPU 필요, 장시간 소요)
- 초기엔 Python lint + 환경 테스트만, 웹 CI는 Phase 3에서 추가

## Artifact 관리

### 모델 파일
- Git에 직접 커밋하지 않음 (수십~수백 MB)
- **HuggingFace Hub** 사용: 모델 버저닝/카드 기능 내장, RL 프로젝트와 궁합 좋음
- `models/checkpoints/`, `models/exported/`는 `.gitignore`에 포함
- 배포용 최종 ONNX/TFJS 모델만 GitHub Releases 또는 `web/`에 포함

### 게임 에셋 (스프라이트, 사운드)
- 파일이 작으므로 Git에 직접 커밋

### Docker
- 현 단계에서는 불필요 (uv + venv로 충분)
- 학습 환경 재현성이 필요해지면 `training/Dockerfile` 추가

### 실험 추적
- W&B 또는 TensorBoard 연동 (SB3 네이티브 지원)
- 하이퍼파라미터, 보상 커브, ELO 추적에 활용

## 코드 복사 방침

- **서브모듈 사용하지 않음** — 두 프로젝트 모두 상당한 커스터마이징 필요
- pika-zoo, pikachu-volleyball 코드를 직접 복사하여 가져옴
- 각 디렉토리에 원본 출처, 라이선스(LICENSE), 변경사항(ATTRIBUTION.md)을 명시

### pika-zoo 환경 코드
- 소스: https://github.com/helpingstar/pika-zoo
- 복사 위치: `training/env/`
- MIT License

### pikachu-volleyball 웹 코드
- 소스: https://github.com/gorisanson/pikachu-volleyball
- 복사 위치: `web/`
- 원본 저장소 라이선스 확인 필요 (UNLICENSED)

## 작업 단계

### Phase 1: 초기 세팅

1. **디렉토리 구조 생성**: 저장소 구조(README.md 참고)대로 생성
2. **pika-zoo 복사**: 환경 코드 전체를 `training/env/`에 배치, LICENSE + ATTRIBUTION.md 포함
3. **pikachu-volleyball 복사**: 전체 소스를 `web/`에 배치, ATTRIBUTION.md 포함
4. **Python 환경**: `pyproject.toml` + `uv lock`으로 의존성 관리
5. **환경 동작 확인**: `training/scripts/test_env.py` — import, reset/step 정상 작동 테스트
6. **기본 PPO 학습 확인**: `training/scripts/train_ppo.py` — PettingZoo → Gymnasium 래퍼 적용 (SB3는 단일 에이전트 인터페이스 필요)

### Phase 2: 학습 고도화

1. Self-play 구현 (과거 버전 상대풀 관리, 상대 교체 전략)
2. PFSP (Prioritized Fictitious Self-Play) 적용
3. 커리큘럼 러닝 (쉬운 환경 → 어려운 환경)
4. ELO 레이팅 추적
5. DuckLL Super AI 대비 벤치마크

### Phase 3: 웹 데모 통합

1. 학습된 PyTorch 모델 → ONNX 변환
2. ONNX → onnxruntime-web 또는 TensorFlow.js 변환
3. 웹 코드에서 규칙 기반 AI 자리에 RL 모델 추론 삽입
4. 로컬 웹에서 사람 vs RL 에이전트 대전 확인
5. GitHub Pages 등으로 온라인 배포
6. (선택) P2P 온라인 대전 통합 (pikachu-volleyball-p2p-online 기반)

## hankluo6 선행 연구 참고

hankluo6은 gym-pikachu-volleyball 기반으로 4가지 조합을 실험:

| 방법 | 규칙AI 상대 평균 점수 (만점 1.0) |
|------|------|
| PPO | 0.71 ± 0.70 |
| PPO (Self-Play) | 0.14 ± 0.99 |
| ES | 0.34 ± 0.94 |
| ES (Self-Play) | -0.38 ± 0.84 |
| Random Policy | -0.87 ± 0.50 |

- PPO가 규칙AI 상대 승률 약 80%로 가장 우수
- Self-play 결과가 상대적으로 약함 → 개선 여지가 큼
- 환경에 랜덤 초기 공 위치/속도 도입 시 학습 난이도 증가
- 엔트로피 계수 추가 시 성능 향상 확인

## 라이선스 참고

- pika-zoo: MIT License
- pikachu-volleyball: 원본 저장소 라이선스 확인 필요 (UNLICENSED)
- 원작 게임 에셋: "(C) SACHI SOFT / SAWAYAKAN Programmers", "(C) Satoshi Takenouchi" 1997
