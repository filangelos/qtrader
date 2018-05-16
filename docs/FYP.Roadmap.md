# Roadmap

## Environment

### Framework

- [x] **observation**: prices vector
- [ ] **reward**: modular
    * [ ] n-step learning
    * [ ] sharpe/shortino ratio
    * [ ] risk-aversion (parameterized by alpha)
    * [ ] averaged return \bar{G}_{t} (instead of discounted return)
- [x] **project**: folder structure refinements

### Misc

- [ ] **`render`**: GUI
    * [x] PnL
    * [ ] Trading Signals
    * [x] Drawdown
    * [x] Sector Risk & Exposures
    * [ ] Asset Risk & Exposures
- [x] **multi-agent**: trading accounts

## Agents

### Pre-Training

- [x] **`tangency-portfolio`**: with transaction costs & random initial portfolio
- [ ] **Q**: meaningful interpretation of action-value function

### Trading Signals

- [ ] **covariance**: volatility predictor

## Market Simulation

- [x] **`surrogates`**: amplitude adjusted fourier transform baseline model
- [x] **`VAR`**: vector autoregressive baseline model
- [ ] **`VAE`**: vanilla variational autoencoder architecure
- [ ] **`GAN`**: vanilla generative adversarial network
- [ ] **evaluation criteria**:
    - [x] **`Moments`**: statistical moments on raw & rolling data
    - [x] statistical arbitrage of generated market
        - [x] quadratic programming
        - [x] random agent
        - [ ] reinforcement agent (pre-trained)
    - [ ] sector classification & statistics

---

## Schedule

### [26.03.18 - 01.04.18 | Week 1](../log/week_1.ipynb)

- [x] **Pre-Training**: `tangency-portfolio` data generator
- [x] **Market Simulation**: baseline models
- [x] **Environment**: observation & OpenAI new API

### [02.04.18 - 08.04.18 | Week 2](../log/week_2.ipynb)

- [x] **Trading Periods**: fix trading periods
- [ ] **Pre-Training**: `tangency-portfolio` with fixed \alpha & sharpe ratio implementations
- [x] **Project**: folder structure refinements

### [09.04.18 - 15.04.18 | Week 3](../log/week_3.ipynb)

- [ ] **Q**: meaningful interpretation of action-value function
- [x] **`render`**: GUI

### [16.04.18 - 22.04.18 | Week 4](../log/week_4.ipynb)

- [x] **Multi-agent**: trading accounts
- [ ] **Reward**: modular

### [23.04.18 - 29.04.18 | Week 5](../log/week_5.ipynb)

- [ ] **Covariance**: volatility predictor

### [30.04.18 - 06.05.18 | Week 6](../log/week_6.ipynb)

### [07.05.18 - 13.05.18 | Week 7](../log/week_7.ipynb)

### [14.05.18 - 20.05.18 | Week 8](../log/week_8.ipynb)

- [ ] **evaluation criteria**: market simulation
    - [x] **`Moments`**: statistical moments on raw & rolling data
    - [x] statistical arbitrage of generated market
        - [x] quadratic programming
        - [x] random agent
- [ ] **`VAE`**: vanilla variational autoencoder architecure
- [ ] **`GAN`**: vanilla generative adversarial network

### [21.05.18 - 27.05.18 | Week 9](../log/week_9.ipynb)

### [28.05.18 - 03.06.18 | Week 10](../log/week_10.ipynb)

### [04.06.18 - 10.06.18 | Week 11](../log/week_11.ipynb)

### [11.06.18 - 17.06.18 | Week 12](../log/week_12.ipynb)

### [18.06.18 - 24.06.18 | Week 13](../log/week_13.ipynb)