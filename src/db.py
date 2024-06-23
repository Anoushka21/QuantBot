import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, JSON, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('mysql+mysqlconnector://root:admin@localhost:3306/quantbot')

Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

class OKXAccount(Base):
    __tablename__ = 'okx_accounts'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    scheduled_timestamp = Column(DateTime)
    data = Column(JSON)
    validated = Column(Boolean, default=False)

class Order(Base):
    __tablename__ = 'orders'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    oems_algorithm_id = Column(Integer)
    order_id = Column(String(255))
    status = Column(String(50))
    data = Column(JSON)
    validated = Column(Boolean, default=False)
    created_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Metrics(Base):
    __tablename__ = 'metrics'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exchange = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    scheduled_timestamp = Column(DateTime)
    portfolio_account_metrics = Column(JSON)
    portfolio_return_metrics = Column(JSON)
    portfolio_risk_metrics = Column(JSON)
    position_execution_quality_metrics = Column(JSON)
    position_risk_metrics = Column(JSON)
    timeframe = Column(String(50))

Base.metadata.create_all(engine)

okx_account_data = [
    OKXAccount(id='1', created_at=datetime(2023, 1, 1, 0, 0, 0), scheduled_timestamp=datetime(2023, 1, 1, 0, 0, 0), validated=True, data={
        "adjEq": '0.0', "borrowFroz": '0.0', "details": [
            {"ccy": 'BTC', "eq": '1.0', "eqUsd": '30000', "availBal": '0.0', "availEq": '0.0', "borrowFroz": '0.0',
             "cashBal": '0.0', "clSpotInUseAmt": '0.0', "crossLiab": '0.0', "disEq": '0.0', "fixedBal": '0.0',
             "frozenBal": '0.0', "imr": '0.0', "interest": '0.0', "isoEq": '0.0', "isoLiab": '0.0', "isoUpl": '0.0',
             "liab": '0.0', "maxLoan": '0.0', "mgnRatio": '0.0', "mmr": '0.0', "notionalLever": '0.0', "ordFrozen": '0.0',
             "rewardBal": '0.0', "smtSyncEq": '0.0', "spotInUseAmt": '0.0', "spotIsoBal": '0.0', "stgyEq": '0.0',
             "twap": '0.0', "uTime": None, "upl": '0.0', "uplLiab": '0.0'},
            {"ccy": 'USDT', "eq": '100000', "eqUsd": '100000', "availBal": '0.0', "availEq": '0.0', "borrowFroz": '0.0',
             "cashBal": '0.0', "clSpotInUseAmt": '0.0', "crossLiab": '0.0', "disEq": '0.0', "fixedBal": '0.0',
             "frozenBal": '0.0', "imr": '0.0', "interest": '0.0', "isoEq": '0.0', "isoLiab": '0.0', "isoUpl": '0.0',
             "liab": '0.0', "maxLoan": '0.0', "mgnRatio": '0.0', "mmr": '0.0', "notionalLever": '0.0', "ordFrozen": '0.0',
             "rewardBal": '0.0', "smtSyncEq": '0.0', "spotInUseAmt": '0.0', "spotIsoBal": '0.0', "stgyEq": '0.0',
             "twap": '0.0', "uTime": None, "upl": '0.0', "uplLiab": '0.0'}
        ], "imr": '0.0', "isoEq": '0.0', "mgnRatio": '0.0', "mmr": '0.0', "notionalUsd": '0.0', "ordFroz": '0.0',
        "totalEq": '130000', "uTime": '0', "upl": '0.0'
    }),
]

order_data = [
    Order(id='1', oems_algorithm_id=0, order_id='1556935276122767360', status='filled', data={
        "live": {"instType": "SPOT", "instId": "MATIC-USDT", "tgtCcy": "", "ccy": "",
                 "ordId": "1556935276122767360", "clOrdId": "17189026861", "algoClOrdId": "", "algoId": "",
                 "tag": "", "px": "0.576", "sz": "1", "notionalUsd": "0.57580992", "ordType": "limit",
                 "side": "buy", "posSide": "", "tdMode": "cash", "accFillSz": "0", "fillNotionalUsd": "",
                 "avgPx": "0", "state": "live", "lever": "0", "pnl": "0", "feeCcy": "MATIC", "fee": "0",
                 "rebateCcy": "USDT", "rebate": "0", "category": "normal", "uTime": "1718902687035",
                 "cTime": "1718902687035", "source": "", "reduceOnly": "false", "cancelSource": "",
                 "quickMgnType": "", "stpId": "", "stpMode": "cancel_maker", "attachAlgoClOrdId": "",
                 "lastPx": "0.5747", "isTpLimit": "false", "slTriggerPx": "", "slTriggerPxType": "",
                 "tpOrdPx": "", "tpTriggerPx": "", "tpTriggerPxType": "", "slOrdPx": "", "fillPx": "",
                 "tradeId": "", "fillSz": "0", "fillTime": "", "fillPnl": "0", "fillFee": "0", "fillFeeCcy": "",
                 "execType": "", "fillPxVol": "", "fillPxUsd": "", "fillMarkVol": "", "fillFwdPx": "",
                 "fillMarkPx": "", "amendSource": "", "reqId": "", "amendResult": "", "code": "0", "msg": "",
                 "pxType": "", "pxUsd": "", "pxVol": "", "linkedAlgoOrd": {"algoId": ""}, "attachAlgoOrds": []},
        "filled": {"instType": "SPOT", "instId": "MATIC-USDT", "tgtCcy": "", "ccy": "",
                   "ordId": "1556935276122767360", "clOrdId": "17189026861", "algoClOrdId": "",
                   "algoId": "", "tag": "", "px": "0.576", "sz": "1", "notionalUsd": "0.57580992",
                   "ordType": "limit", "side": "buy", "posSide": "", "tdMode": "cash", "accFillSz": "1",
                   "fillNotionalUsd": "0.574310415", "avgPx": "0.5745", "state": "filled", "lever": "0",
                   "pnl": "0", "feeCcy": "MATIC", "fee": "-0.001", "rebateCcy": "USDT", "rebate": "0",
                   "category": "normal", "uTime": "1718902687036", "cTime": "1718902687035", "source": "",
                   "reduceOnly": "false", "cancelSource": "", "quickMgnType": "", "stpId": "",
                   "stpMode": "cancel_maker", "attachAlgoClOrdId": "", "lastPx": "0.5747",
                   "isTpLimit": "false", "slTriggerPx": "", "slTriggerPxType": "", "tpOrdPx": "",
                   "tpTriggerPx": "", "tpTriggerPxType": "", "slOrdPx": "", "fillPx": "0.5745",
                   "tradeId": "6176408", "fillSz": "1", "fillTime": "1718902687035", "fillPnl": "0",
                   "fillFee": "-0.001", "fillFeeCcy": "MATIC", "execType": "T", "fillPxVol": "",
                   "fillPxUsd": "", "fillMarkVol": "", "fillFwdPx": "", "fillMarkPx": "", "amendSource": "",
                   "reqId": "", "amendResult": "", "code": "0", "msg": "", "pxType": "", "pxUsd": "",
                   "pxVol": "", "linkedAlgoOrd": {"algoId": ""}, "attachAlgoOrds": []}}),
]


metrics_data = [
    Metrics(id='1', exchange='OK', created_at=datetime(2023, 6, 1, 12, 0, 0),
            scheduled_timestamp=datetime(2023, 12, 1, 12, 0, 0), portfolio_account_metrics={
                "net_asset_value": 150000.00, "assets": 140000.00, "cash": 10000.00, "margin": 5000.00,
                "buying_power": 20000.00, "margin_utilization": 25.00, "margin_available": 15000.00,
                "volume": 0, "turnover": 0, "number_of_trades": 0},
            portfolio_return_metrics={"unrealized_pnl": 5000.00, "realized_pnl": 2000.00,
                                      "avg_pnl_per_trade": 100.00, "avg_positive_pnl": 150.00,
                                      "avg_negative_pnl": -50.00, "fees": 300.00, "cagr": 10.00,
                                      "win_rate": 60.00, "up_time": 70.00, "down_time": 30.00,
                                      "total_trades": 20, "long_trades": 15, "short_trades": 5},
            portfolio_risk_metrics={"concentration": 30.00, "value_at_risk": 10000.00, "volatility": 5.00,
                                    "standard_deviation": 4.50, "beta": 1.20, "sharpe": 1.50,
                                    "max_drawdown": 20.00, "drawdown_period": 30, "drawdown_recovery": 20,
                                    "max_runup": 25.00, "runup_period": 15},
            position_execution_quality_metrics={"unrealized_pnl": 3000.00, "realized_pnl": 1000.00, "fees": 150.00},
            position_risk_metrics={"concentration": 20.00, "value_at_risk": 5000.00, "sharpe_ratio": 1.30,
                                   "beta": 0.90, "volatility": 4.00, "standard_deviation": 4.00,
                                   "max_drawdown": 15.00, "drawdown_period": 20, "drawdown_recovery": 10,
                                   "max_runup": 18.00, "runup_period": 12},
            timeframe="2023-06-01 to 2023-12-01"),
]

for data in okx_account_data:
    session.add(data)
session.commit()

for data in order_data:
    session.add(data)
session.commit()

for data in metrics_data:
    session.add(data)
session.commit()
