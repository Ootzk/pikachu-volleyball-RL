# pikachu-volleyball-RL

> Train a Pikachu Volleyball AI with reinforcement learning and play against it in your browser

피카츄배구(1997) 리버스 엔지니어링 기반의 강화학습 에이전트를 훈련하고, **브라우저에서 직접 대전 가능한 웹 데모**를 제공하는 프로젝트.

기존 피카츄배구 RL 프로젝트들은 Python 환경에서 학습/평가까지만 수행하고 끝나지만, 이 프로젝트는 **학습된 에이전트를 브라우저에서 사람이 직접 상대**할 수 있고, **P2P 온라인 대전**까지 지원하는 것이 목표.

## 기존 프로젝트 관계도

| 저장소 | 설명 | 역할 |
|--------|------|------|
| [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) | 원작 리버스 엔지니어링 → JS 재구현 (물리엔진+규칙AI, PixiJS 렌더링) | 참고 (웹 버전의 원본) |
| [gorisanson/pikachu-volleyball-p2p-online](https://github.com/gorisanson/pikachu-volleyball-p2p-online) | 위 프로젝트에 WebRTC P2P 온라인 대전 추가 | **웹 데모 베이스로 코드 복사** |
| [helpingstar/pika-zoo](https://github.com/helpingstar/pika-zoo) | 피카츄배구 물리엔진을 Python으로 포팅, PettingZoo 멀티에이전트 RL 환경 | **학습 환경 베이스로 코드 복사** |
| [DuckLL/pikachu-volleyball](https://github.com/duckll/pikachu-volleyball) | gorisanson 포크, 규칙 기반 AI 강화 (슈퍼서브, 예측공격 등) | 벤치마크 상대로 참고 |
| [hankluo6/Pikachu-VolleyBall-RL](https://github.com/hankluo6/Pikachu-VolleyBall-RL) | PPO/ES + Self-play로 학습 (gym 기반). 웹 데모 없음 | 선행 연구 참고 |

## 기술 스택

- **학습 환경**: pika-zoo (PettingZoo 멀티에이전트 인터페이스)
- **학습 프레임워크**: PyTorch + Stable Baselines3 (SB3)
- **학습 알고리즘**: PPO → Self-play (PFSP 등 고급 기법 적용 예정)
- **모델 변환**: PyTorch → ONNX → onnxruntime-web 또는 TensorFlow.js
- **웹 데모**: gorisanson/pikachu-volleyball-p2p-online 기반 (JavaScript, PixiJS, WebRTC)

## 저장소 구조 (모노레포)

```
pikachu-volleyball-RL/
├── training/                # Python 학습 파이프라인
│   ├── env/                 # pika-zoo 환경 코드 복사
│   ├── scripts/             # 학습 스크립트 (train_ppo.py, train_selfplay.py 등)
│   └── configs/             # 하이퍼파라미터 설정
├── models/                  # 학습된 모델 저장
│   ├── checkpoints/         # PyTorch 체크포인트
│   └── exported/            # 변환된 ONNX / TFJS 모델
├── scripts/                 # 모델 변환 스크립트 (PyTorch → ONNX → Web)
├── web/                     # JavaScript 웹 데모
│   └── (p2p-online 코드 복사 + RL 에이전트 통합)
├── docs/                    # 문서
├── LICENSE
└── README.md
```

## 작업 단계

### Phase 1: 초기 세팅 (현재)
1. 저장소 디렉토리 구조 생성
2. pika-zoo 환경 코드를 `training/env/`에 복사
3. p2p-online 웹 코드를 `web/`에 복사
4. Python 환경 세팅 (requirements.txt)
5. pika-zoo 환경 동작 확인
6. SB3 PPO로 규칙 기반 AI 상대 기본 학습 동작 확인

### Phase 2: 학습 고도화
1. Self-play 구현 (과거 버전 상대풀 관리, 상대 교체 전략)
2. PFSP (Prioritized Fictitious Self-Play) 적용
3. 커리큘럼 러닝 (쉬운 환경 → 어려운 환경)
4. ELO 레이팅 추적
5. DuckLL Super AI 대비 벤치마크

### Phase 3: 웹 데모 통합
1. 학습된 PyTorch 모델 → ONNX 변환
2. ONNX → onnxruntime-web 또는 TensorFlow.js 변환
3. p2p-online 웹 코드에서 규칙 기반 AI 자리에 RL 모델 추론 삽입
4. 로컬 웹에서 사람 vs RL 에이전트 대전 확인
5. P2P 온라인 대전에 RL 에이전트 통합
6. GitHub Pages 등으로 온라인 배포

## 라이선스

- pika-zoo: MIT License
- pikachu-volleyball-p2p-online: 원본 저장소 라이선스 확인 필요
- 원작 게임 에셋: "(C) SACHI SOFT / SAWAYAKAN Programmers", "(C) Satoshi Takenouchi" 1997