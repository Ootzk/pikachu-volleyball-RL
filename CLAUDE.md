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

### 실험 관리 (`experiments/`)
- `experiments/` 전체를 `.gitignore`로 제외 (로컬 전용)
- 실험 단위로 **자립형 폴더** 관리
- 네이밍 규칙:
  - 학습 실험: `{번호}_train_{대상}_with_{상대}` (예: `001_train_p1_with_builtin`)
  - 평가 실험: `{번호}_eval_{대상1}_vs_{대상2}` (예: `003_eval_001_vs_002`)
  - Self-play: `{번호}_selfplay_{설명}` (예: `007_selfplay_curriculum`)
- 폴더 내부 구조:
  ```
  experiments/007_selfplay_curriculum/
  ├── README.md             # 설정, 결과, 교훈
  ├── run.sh                # 실행 스크립트 (재현용)
  ├── curriculum.json        # 커리큘럼 설정 (해당 시)
  ├── checkpoints/p1/, p2/  # 상대풀 + 체크포인트 통합
  ├── tensorboard/          # TensorBoard 로그
  ├── model_p1.zip          # 최종 모델
  ├── model_p2.zip
  └── *.mp4                 # 대전 영상
  ```
- 이전 실험 이어받기: `cp -r experiments/007 experiments/009` → `run.sh` 수정
- **절대 다른 실험의 산출물을 `rm -rf`로 삭제하지 않음**
- TensorBoard: `uv run tensorboard --logdir experiments/`로 전체 비교
- 배포용 최종 ONNX/TFJS 모델은 GitHub Releases 또는 `web/`에 포함

### 영상 녹화
- `training/scripts/record_episode.py`로 에피소드 MP4 생성 (ffmpeg 스트리밍, 메모리 절약)
- 실험 폴더 내 `{p1}_vs_{p2}.mp4` 형식으로 보관
- README.md에 `<video>` 태그로 첨부 (VSCode 미리보기 지원)

### 게임 에셋 (스프라이트, 사운드)
- 파일이 작으므로 Git에 직접 커밋

### Docker
- 현 단계에서는 불필요 (uv + venv로 충분)
- 학습 환경 재현성이 필요해지면 `training/Dockerfile` 추가

## 학습 아키텍처

### 이중 모델 (p1/p2 분리)
- 물리 엔진의 좌우 비대칭 때문에 단일 모델 불가
  - 비대칭 1: 공의 벽 반사 경계가 좌/우 다름 (physics.py 393~403)
  - 비대칭 2: 파워히트 방향이 위치 기반 (physics.py 622~626)
  - 비대칭 3: p1에만 `down_right_key` 존재 (SimplifyAction 래퍼로 추상화됨)
- p1_model: 항상 player_1 (좌측)
- p2_model: 항상 player_2 (우측)

### Self-play 구조
- `train_selfplay.py`: p1/p2 교대 학습
- 상대 비율: `latest_prob` + `builtin_prob` + `pool(PFSP)`
- `--curriculum curriculum.json`으로 동적 상대 비율 지정 (선형 보간)
- PFSP: 승률 낮은 상대를 우선 샘플링
- 평가: 5매치업 병렬 실행 (`ProcessPoolExecutor`, `spawn` 컨텍스트)

### TensorBoard 로거 구조
- `p1` 로거: p1 train 메트릭 + p1 관점 eval (vs_builtin, vs_random, vs_p2)
- `p2` 로거: p2 train 메트릭 + p2 관점 eval (vs_builtin, vs_random, vs_p1)
- `common` 로거: 커리큘럼 메타데이터 (builtin_prob, latest_prob, pool_prob)
- Custom Scalars: 승률 비교, 득점 비교, 커리큘럼 비율을 오버레이 차트로 제공

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

### Phase 2: 학습 고도화 (진행 중)

1. ~~Self-play 구현~~ — 완료 (교대 학습, PFSP 상대풀, builtin 앵커)
2. ~~PFSP 적용~~ — 완료 (OpponentPool, 승률 기반 가중 샘플링)
3. ~~커리큘럼 러닝~~ — 완료 (JSON 기반 동적 상대 비율)
4. ~~ELO/평가 시스템~~ — 완료 (evaluate.py, 서브별 통계, 평균 득점)
5. **장기 학습** — 진행 중 (007에서 builtin 55%, 추가 학습 필요)
6. DuckLL Super AI 포팅/벤치마크 — 미착수

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
- Self-play 결과가 상대적으로 약함 (0.14) → 커리큘럼/builtin 앵커 없이 순수 self-play의 한계
- 환경에 랜덤 초기 공 위치/속도 도입 시 학습 난이도 증가
- 엔트로피 계수 추가 시 성능 향상 확인
- 커리큘럼(builtin 앵커) + ent_coef 조합으로 self-play 성능 개선 가능

## 라이선스 참고

- pika-zoo: MIT License
- pikachu-volleyball: 원본 저장소 라이선스 확인 필요 (UNLICENSED)
- 원작 게임 에셋: "(C) SACHI SOFT / SAWAYAKAN Programmers", "(C) Satoshi Takenouchi" 1997
