# Roadmap

## Library

- [x] **project**: folder structure refinements
- [x] **`contrib`**: temporary/untested code
    * [x] `rl`: reinforcement-learning agents
- [x] **`setup.py`**: PyPI-friendly
    * [x] `requirements.txt`: development only
    * [x] tests

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
- [ ] **runner**: automated execution
    * [ ] fixed agent API
    * [ ] multiple episodes

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

- [ ] **base**: common interface
    * [ ] `fit` method
    * [ ] private API (agent-specific)
    * [ ] public API (environment runner friendly)

### evaluation-metrics

- [ ] **synthetic market**: dummy (noisy) data
    * [ ] sine waves
    * [ ] sawtooth waves
    * [ ] chirp waves
- [ ] **S&P500**: 5 years data
    * [ ] handful universe
    * [ ] whole market

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

- [ ] **persistance**: one-step look back
- [ ] **ARIMA**: autoregressive integrated moving average environment model
- [ ] **RNN**: recurrent neural network environment model

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

- [ ] **AAFT**: amplitude adjusted fourier transform baseline model
- [ ] **VAR**: vector autoregressive baseline model

### generative-models

- [ ] **VAE**: vanilla variational autoencoder architecure
    * [ ] fixed size sequence generation
    * [ ] recurrent architecture
    * [ ] comparison to baselines
- [ ] **GAN**: vanilla generative adversarial network
    * [ ] fixed size sequence generation
    * [ ] recurrent architecture
    * [ ] comparison to baselines