# Roadmap

## Library

- [x] **project**: folder structure refinements
- [x] **`contrib`**: temporary/untested code
    * [x] `rl`: reinforcement-learning agents
- [x] **`setup.py`**: PyPI-friendly
    * [x] `requirements.txt`: development only
    * [x] tests
- [ ] **report**: strategy/agent comparison summary
    * `plotting`: PnL, trades, drawdown
    * `stats`: PnL, sharpe ratio, hit rate, adjusted metrics

## Environment

### API

- [x] **observation**: `dict` <- `numpy.ndarray`
    * [x] prices vector
    * [x] returns vector
    * [x] last weights
- [ ] **reward**: modular architecture
    * [x] n-step learning: varying `trading_period`
    * [ ] risk-aversion (parameterized by alpha)
    * [ ] sharpe/shortino ratio
    * [ ] averaged return \bar{G}_{t} (instead of discounted return)
- [x] **multi-agent**: trading accounts
    * [x] rewards `pandas.DataFrame`
    * [x] actions `pandas.DataFrame`
- [x] **runner**: automated execution
    * [x] fixed agent API
    * [x] multiple episodes

### GUI

- [ ] **summary**: live calculations
    * [x] PnL
    * [ ] drawdown
    * [ ] asset risk
    * [ ] sector risk
- [x] **trading-signal**: strategies
    * [x] scatter: varying-intensity

## Agent

### API

- [x] **base**: common interface
    * [x] `fit` method
    * [x] private API (agent-specific)
    * [x] public API (environment runner friendly)

### evaluation-metrics

- [ ] **synthetic market**: dummy (noisy) data
    * [ ] sine waves
    * [ ] sawtooth waves
    * [ ] chirp waves
- [x] **S&P500**: 5 years data
    * [x] handful universe
    * [x] whole market

### baseline models

- [x] **random**: random selection of allocations
- [x] **uniform**: 1/N allocation

### pre-training

- [x] **quadratic-programming**: general optimizer
    * [x] general objective function
    * [x] supervised data generator
    * [x] trading agent
- [ ] **Q**: meaningful interpretation of action-value function

### model-base

- [x] **persistence**: one-step look back
- [x] **VAR**: vector autoregressive environment model
- [x] **RNN**: recurrent neural network environment model

### model-free

- [ ] **DQN**: binary trader
- [ ] **DDPG**: portfolio allocator

## Market-Simulation

### evaluation-metrics

- [ ] **moments**: statistical moments on raw & rolling data
- [ ] **arbitrage**: statistical arbitrage of generated market
    * [ ] quadratic programming
    * [ ] random agent
    * [ ] reinforcement agent
- [ ] **sector**: classification & statistics

### baseline-models

- [x] **AAFT**: amplitude adjusted fourier transform baseline model
- [x] **VAR**: vector autoregressive baseline model

### generative-models

- [ ] **VAE**: vanilla variational autoencoder architecure
    * [ ] fixed size sequence generation
    * [ ] recurrent architecture
    * [ ] comparison to baselines
- [ ] **GAN**: vanilla generative adversarial network
    * [ ] fixed size sequence generation
    * [ ] recurrent architecture
    * [ ] comparison to baselines