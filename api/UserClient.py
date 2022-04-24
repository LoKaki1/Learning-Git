"""

Class to control the user data such as tokens, passwords, watchlist and scanners and all their settings
I'd save in the future...
The user database will look like this:
    Username: {
        token:
        hash_password:
        data: {
            Watchlist:   in the future I'd like to save the watchlist with labels such as in yahoo.
            [
                {
                    Ticker:, predicted_price:, last_price, prestange_of_gap, ?ratio_of_sucssess (not sure about this)
                }...;
            ]
            Scanner:   in the future I'd like to add a way to control scanners now I'm not so good about it.
            [
                {
                    Ticker, price, last_price, gap, presnatage_gap, volume
                }
            ]
            ?Settings: not sure if I add a way to control settings of AI from the frontend
             [
                {
                    Ticker: , **settings(epochs, prediction_days, units, predcition_day, **layers,)
                } ...;
             ]
        }
    }


"""
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class UserClient:
    username: str
    token: str
    hash_password: str
    data: dict


user = UserClient("shahar", '12', '#@', {
    })
print(user)
