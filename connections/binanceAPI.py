import hashlib
import hmac
import json
import time
import os
from datetime import datetime, timedelta
from urllib.parse import urlencode
import pandas as pd
import requests
import websocket
from utils.config import BINANCE_API_SECRET, BINANCE_API_KEY
from data.db import DatabaseManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class BinanceClient:
    def __init__(self):
        self.api_key = BINANCE_API_KEY
        self.api_secret = BINANCE_API_SECRET
        self.base_url = "https://api.binance.com"
        self.base_url_v3 = f"{self.base_url}/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        })
        self.active_websockets = {}
        self.db_manager = DatabaseManager()
        self.reconnect_required = False
        self.supported_symbols = self.db_manager.supported_symbols
        self.cache = {
            'ticker_price': {},
            'order_book': {},
            'klines': {}
        }
        self.cache_ttl = {
            'ticker_price': 5,
            'order_book': 2,
            'klines': 60
        }
        self.last_cache_update = {
            'ticker_price': {},
            'order_book': {},
            'klines': {}
        }

    # Генерація цифрового підпису
    def _generate_signature(self, params):
        """
            Генерує цифровий підпис для захищеного запиту до API Binance.

            Параметри:
                params (dict): Словник параметрів, які будуть використані для створення підпису.

            Повертає:
                str: HMAC SHA256 підпис у шістнадцятковому форматі.
            """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    # Перевірка допустимості символу
    def _validate_symbol(self, symbol):
        """
           Перевіряє, чи входить вказаний торговий символ до списку підтримуваних.

           Параметри:
               symbol (str): Торговий символ (наприклад, 'BTCUSDT').

           Повертає:
               Tuple[bool, Optional[str]]: Кортеж, де перше значення — це результат перевірки (True/False),
               а друге — базовий символ (наприклад, 'BTC') або None, якщо символ не підтримується.
           """
        base_symbol = None

        # Витягуємо базовий символ (наприклад, BTC з BTCUSDT)
        for s in self.supported_symbols:
            if symbol.startswith(s):
                base_symbol = s
                break

        if not base_symbol:
            self.db_manager.log_event('WARNING', f"Непідтримувана криптовалюта: {symbol}", 'BinanceClient')
            return False, None

        return True, base_symbol

    # Отримання даних про ціну у вигляді свічок
    def get_klines(self, symbol, timeframe, limit=100, start_time=None, end_time=None, use_cache=True):
        """
            Отримує історичні дані про ціни у вигляді японських свічок (Klines) з API Binance.

            Параметри:
                symbol (str): Торговий символ (наприклад, 'BTCUSDT').
                timeframe (str): Інтервал свічок (наприклад, '1m', '1h', '1d').
                limit (int, optional): Кількість свічок для отримання. За замовчуванням 100.
                start_time (int, optional): Початковий час у мілісекундах з UNIX-епохи.
                end_time (int, optional): Кінцевий час у мілісекундах з UNIX-епохи.
                use_cache (bool, optional): Чи використовувати кешовані дані. За замовчуванням True.

            Повертає:
                pd.DataFrame: Таблиця з колонками:
                    - 'open_time', 'open', 'high', 'low', 'close', 'volume',
                    - 'close_time', 'quote_asset_volume', 'number_of_trades',
                    - 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                У разі помилки — повертає порожній DataFrame.
            """
        cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}_{limit}"

        if use_cache and cache_key in self.cache['klines']:
            cache_time = self.last_cache_update['klines'].get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl['klines']:
                return self.cache['klines'][cache_key]

        endpoint = f"{self.base_url_v3}/klines"
        params = {
            'symbol': symbol,
            'interval': timeframe,  # FIXED: changed 'timeframe' to 'interval' for Binance API
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting klines: {e}", 'BinanceClient')
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        if df.empty:
            return df

        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Оновлюємо кеш
        if use_cache:
            self.cache['klines'][cache_key] = df.copy()
            self.last_cache_update['klines'][cache_key] = time.time()

        return df

    # Отримання поточної ціни
    def get_ticker_price(self, symbol=None, use_cache=True):
        """
           Отримує поточну ринкову ціну (ticker price) для заданого торгового символу або для всіх символів.

           Параметри:
               symbol (str, optional): Торговий символ (наприклад, 'BTCUSDT'). Якщо None, повертає ціни для всіх символів.
               use_cache (bool, optional): Чи використовувати кеш. За замовчуванням True.

           Повертає:
               Union[dict, list]: Дані з ціною для одного символу (dict) або список усіх символів з цінами (list).
               У разі помилки — порожній словник.
           """
        if use_cache and symbol and symbol in self.cache['ticker_price']:
            cache_time = self.last_cache_update['ticker_price'].get(symbol, 0)
            if time.time() - cache_time < self.cache_ttl['ticker_price']:
                return self.cache['ticker_price'][symbol]

        endpoint = f"{self.base_url_v3}/ticker/price"
        params = {}
        if symbol:
            params['symbol'] = symbol

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Оновлюємо кеш
            if use_cache:
                if symbol:
                    self.cache['ticker_price'][symbol] = data
                    self.last_cache_update['ticker_price'][symbol] = time.time()
                else:
                    # Якщо запитуються всі ціни, оновлюємо кеш для кожного символу
                    for item in data:
                        self.cache['ticker_price'][item['symbol']] = {
                            'symbol': item['symbol'],
                            'price': item['price']
                        }
                        self.last_cache_update['ticker_price'][item['symbol']] = time.time()

            return data
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting ticker price: {e}", 'BinanceClient')
            return {}

    # Отримання останніх угод
    def get_recent_trades(self, symbol, limit=100):
        """
            Отримує останні угоди (trades) для заданого торгового символу.

            Параметри:
                symbol (str): Торговий символ (наприклад, 'BTCUSDT').
                limit (int, optional): Кількість угод, які потрібно отримати. Максимум — 1000. За замовчуванням 100.

            Повертає:
                pd.DataFrame: Таблиця з інформацією про останні угоди, включаючи:
                    - 'id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch'
                У разі помилки — повертає порожній DataFrame.
            """
        endpoint = f"{self.base_url_v3}/trades"
        params = {
            'symbol': symbol,
            'limit': limit
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting recent trades: {e}", 'BinanceClient')
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            return df

        numeric_columns = ['price', 'qty', 'quoteQty']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        df['time'] = pd.to_datetime(df['time'], unit='ms')

        return df

    def get_24hr_ticker_statistics(self, symbol=None):
        """
            Отримує статистику по символу за останні 24 години, або по всіх символах.

            Параметри:
                symbol (str, optional): Торговий символ (наприклад, 'ETHUSDT'). Якщо None, повертає статистику для всіх символів.

            Повертає:
                Union[dict, list]: Статистика у вигляді словника (для одного символу) або списку словників (для всіх символів).
                У разі помилки — повертає порожній словник.
            """
        endpoint = f"{self.base_url_v3}/ticker/24hr"
        params = {}
        if symbol:
            params['symbol'] = symbol

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.db_manager.log_event('ERROR', f"Error getting 24hr ticker statistics: {e}", 'BinanceClient')
            return {}

    # ===== WebSocket methods for real-time data =====

    def _save_kline_to_db(self, kline_data):
        """
            Зберігає один запис японської свічки (kline) у базу даних.

            Параметри:
                kline_data (dict): Дані свічки у форматі, отриманому з WebSocket або API Binance.
                    Очікувані ключі: 'symbol', 'timeframe', 'open_time', 'open', 'high', 'low',
                    'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_volume', 'taker_buy_quote_volume', 'is_closed'.

            Повертає:
                bool: True, якщо збереження пройшло успішно, False у разі помилки або недопустимого символу.
            """
        try:
            # Отримуємо базову валюту з символа (напр. BTC з BTCUSDT)
            symbol = kline_data['symbol']
            is_valid, crypto_symbol = self._validate_symbol(symbol)

            if not is_valid:
                return False

            # Перетворюємо дані у формат, який очікує DatabaseManager
            kline_db_data = {
                'timeframe': kline_data['timeframe'],  # 'timeframe' used correctly for DB
                'open_time': kline_data['open_time'],
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'close_time': kline_data['close_time'],
                'quote_asset_volume': float(kline_data['quote_asset_volume']),
                'number_of_trades': int(kline_data['number_of_trades']),
                'taker_buy_base_volume': float(kline_data['taker_buy_base_volume']),
                'taker_buy_quote_volume': float(kline_data['taker_buy_quote_volume']),
                'is_closed': kline_data['is_closed']
            }

            # Вставка даних у відповідну таблицю
            result = self.db_manager.insert_kline(symbol, kline_db_data)
            return result
        except Exception as e:
            self.db_manager.log_event('ERROR', f"Error saving kline to database: {e}", 'BinanceClient')
            return False


    def start_kline_socket(self, symbol, timeframe, callback, save_to_db=True):
        """
           Запускає WebSocket для підключення до стріму японських свічок (kline) з Binance API.

           Параметри:
               symbol (str): Торговий символ (наприклад, 'BTCUSDT').
               timeframe (str): Інтервал свічок (наприклад, '1m', '5m', '1h').
               callback (Callable): Користувацька функція, яка буде викликатись при кожному новому повідомленні.
               save_to_db (bool, optional): Чи зберігати дані свічок до бази даних. За замовчуванням True.

           Повертає:
               websocket.WebSocketApp: Об'єкт активного WebSocket-з'єднання або None у разі помилки валідації символу.

           Додатково:
               - Дані свічок кешуються у self.active_websockets.
               - У разі втрати з'єднання встановлюється прапорець перепідключення.
               - Інформація про статус WebSocket фіксується у базі через db_manager.
           """
        # Перевіряємо, чи є базовий символ допустимим
        is_valid, _ = self._validate_symbol(symbol)
        if not is_valid:
            self.db_manager.log_event('ERROR', f"Cannot start kline socket for unsupported symbol: {symbol}",
                                      'BinanceClient')
            return None

        socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{timeframe}"

        # Додаємо запис про WebSocket-з'єднання в базу даних
        if save_to_db:
            self.db_manager.update_websocket_status(symbol, 'kline', timeframe, True)

        def on_message(ws, message):
            try:
                callback(ws, message)

                if save_to_db:
                    try:
                        data = json.loads(message)
                        candle = data['k']

                        kline_data = {
                            'symbol': candle['s'],  # symbol
                            'timeframe': candle['i'],  # interval from websocket becomes timeframe in DB
                            'open_time': datetime.fromtimestamp(candle['t'] / 1000),  # open_time
                            'open': candle['o'],  # open
                            'high': candle['h'],  # high
                            'low': candle['l'],  # low
                            'close': candle['c'],  # close
                            'volume': candle['v'],  # volume
                            'close_time': datetime.fromtimestamp(candle['T'] / 1000),  # close_time
                            'quote_asset_volume': candle['q'],  # quote asset volume
                            'number_of_trades': candle['n'],  # number of trades
                            'taker_buy_base_volume': candle['V'],  # taker buy base volume
                            'taker_buy_quote_volume': candle['Q'],  # taker buy quote volume
                            'is_closed': candle['x']  # is closed
                        }

                        self._save_kline_to_db(kline_data)
                    except Exception as e:
                        self.db_manager.log_event('ERROR', f"Error saving kline data to database: {e}", 'BinanceClient')
            except Exception as e:
                self.db_manager.log_event('ERROR', f"Error in on_message handler for kline: {e}", 'BinanceClient')

        def on_error(ws, error):
            self.db_manager.log_event('ERROR', f"WebSocket Error: {error}", 'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', timeframe, False)

        def on_close(ws, close_status_code, close_msg):
            self.db_manager.log_event('INFO',
                                      f"WebSocket Connection Closed: {close_status_code}, {close_msg if close_msg else 'No message'}",
                                      'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = True
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', timeframe, False)
            # Запускаємо пізніше перепідключення
            self.reconnect_required = True

        def on_open(ws):
            self.db_manager.log_event('INFO', f"WebSocket Connection Opened for {symbol} {timeframe} klines",
                                      'BinanceClient')
            ws.custom_reconnect_info['needs_reconnect'] = False
            if save_to_db:
                self.db_manager.update_websocket_status(symbol, 'kline', timeframe, True)

        ws = websocket.WebSocketApp(
            socket_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Додаємо механізм пінгу для підтримки з'єднання
        ws.on_ping = lambda ws, message: ws.send(json.dumps({"type": "pong"}))

        # Додаємо необхідну інформацію для перепідключення
        ws.custom_reconnect_info = {
            'needs_reconnect': False,
            'symbol': symbol,
            'timeframe': timeframe,
            'callback': callback,
            'save_to_db': save_to_db,
            'socket_type': 'kline',
            'last_ping_time': time.time()
        }

        import threading
        ws_thread = threading.Thread(target=ws.run_forever, kwargs={'ping_interval': 30, 'ping_timeout': 10})
        ws_thread.daemon = True
        ws_thread.start()

        # Зберігаємо веб-сокет в активних з'єднаннях
        ws_key = f"kline_{symbol}_{timeframe}"
        self.active_websockets[ws_key] = {
            'ws': ws
        }

        return ws


    # Перевірка та автоматичне перепідключення сокетів
    def check_websocket_connections(self):
        """
            Перевіряє активні WebSocket-з'єднання та виконує автоматичне перепідключення, якщо з'єднання втрачено
            або пінг не було отримано протягом 2 хвилин.

            Визначає проблемні сокети та викликає reconnect_websocket для кожного з них.
            """
        disconnected_sockets = []

        for key, ws_info in list(self.active_websockets.items()):
            ws = ws_info['ws']
            if not ws.sock or not ws.sock.connected or ws.custom_reconnect_info['needs_reconnect']:
                self.db_manager.log_event('WARNING', f"WebSocket {key} is disconnected or requires reconnection.",
                                          'BinanceClient')
                disconnected_sockets.append(key)
            else:
                # Перевіряємо час останнього пінгу
                if 'last_ping_time' in ws.custom_reconnect_info:
                    last_ping = ws.custom_reconnect_info['last_ping_time']
                    if time.time() - last_ping > 120:  # 2 хвилини без пінгу
                        self.db_manager.log_event('WARNING', f"WebSocket {key} has not received ping for too long.",
                                                  'BinanceClient')
                        disconnected_sockets.append(key)

        # Перепідключення відключених сокетів
        for key in disconnected_sockets:
            self.reconnect_websocket(key)

    def reconnect_websocket(self, ws_key):
        """
           Виконує перепідключення до WebSocket за заданим ключем.

           Параметри:
               ws_key (str): Унікальний ключ з'єднання, наприклад 'kline_BTCUSDT_1m'.

           Повертає:
               bool: True, якщо перепідключення виконано успішно, False у разі помилки або невідомого ключа.
           """
        if ws_key not in self.active_websockets:
            self.db_manager.log_event('ERROR', f"Cannot reconnect unknown WebSocket: {ws_key}", 'BinanceClient')
            return False

        # Закриваємо старий WebSocket
        old_ws = self.active_websockets[ws_key]['ws']
        reconnect_info = old_ws.custom_reconnect_info

        try:
            old_ws.close()
        except Exception as e:
            print(f"Error closing old WebSocket {ws_key}: {e}")
            self.db_manager.log_event('ERROR', f"Error closing old WebSocket {ws_key}: {e}", 'BinanceClient')

        # Пауза перед повторним підключенням
        time.sleep(2)

        # Перепідключення в залежності від типу сокета
        if reconnect_info['socket_type'] == 'kline':
            print(f"Reconnecting kline WebSocket for {reconnect_info['symbol']} {reconnect_info['timeframe']}...")
            self.db_manager.log_event('INFO',
                                      f"Reconnecting kline WebSocket for {reconnect_info['symbol']} {reconnect_info['timeframe']}...",
                                      'BinanceClient')
            self.start_kline_socket(
                reconnect_info['symbol'],
                reconnect_info['timeframe'],
                reconnect_info['callback'],
                reconnect_info['save_to_db']
            )

        print(f"Successfully reconnected WebSocket: {ws_key}")
        self.db_manager.log_event('INFO', f"Successfully reconnected WebSocket: {ws_key}", 'BinanceClient')
        return True

    # Зупинка роботи веб сокета
    def close_websocket(self, ws_or_key):
        """
            Закриває одне WebSocket-з'єднання.

            Параметри:
                ws_or_key (str або WebSocketApp): Ключ WebSocket-з'єднання (наприклад, 'kline_BTCUSDT_1m')
                або сам об'єкт WebSocketApp.

            Повертає:
                bool: True, якщо з'єднання було закрито, False — якщо ключ не знайдено або не вдалося завершити з'єднання.
            """
        if isinstance(ws_or_key, str):
            if ws_or_key in self.active_websockets:
                ws_info = self.active_websockets[ws_or_key]
                ws = ws_info['ws']

                socket_parts = ws_or_key.split('_')
                if socket_parts[0] == 'kline':
                    symbol = socket_parts[1]
                    timeframe = socket_parts[2]
                    self.db_manager.update_websocket_status(symbol, 'kline', timeframe, False)

                try:
                    ws.close()
                except Exception as e:
                    print(f"Error closing WebSocket {ws_or_key}: {e}")
                    self.db_manager.log_event('ERROR', f"Error closing WebSocket {ws_or_key}: {e}", 'BinanceClient')
                print(f"Closed WebSocket connection: {ws_or_key}")
                self.db_manager.log_event('INFO', f"Closed WebSocket connection: {ws_or_key}", 'BinanceClient')
                return True
            return False
        else:
            for key, ws_info in self.active_websockets.items():
                if ws_info['ws'] == ws_or_key:
                    try:
                        ws_or_key.close()
                    except Exception as e:
                        print(f"Error closing WebSocket {key}: {e}")
                        self.db_manager.log_event('ERROR', f"Error closing WebSocket {key}: {e}", 'BinanceClient')
                    print(f"Closed WebSocket connection: {key}")
                    self.db_manager.log_event('INFO', f"Closed WebSocket connection: {key}", 'BinanceClient')

                    # Оновлюємо статус у БД
                    socket_parts = key.split('_')
                    if socket_parts[0] == 'kline':
                        symbol = socket_parts[1]
                        timeframe = socket_parts[2]
                        self.db_manager.update_websocket_status(symbol, 'kline', timeframe, False)

                    return True
            return False

    def close_all_websockets(self):
        """
           Закриває всі активні WebSocket-з'єднання.

           Оновлює статуси з'єднань у базі даних та очищає список активних WebSocket з'єднань.
           """
        for key, ws_info in list(self.active_websockets.items()):
            try:
                ws_info['ws'].close()
                print(f"Closed WebSocket connection: {key}")
                self.db_manager.log_event('INFO', f"Closed WebSocket connection: {key}", 'BinanceClient')

                # Оновлюємо статус у БД
                socket_parts = key.split('_')
                if socket_parts[0] == 'kline':
                    symbol = socket_parts[1]
                    timeframe = socket_parts[2]
                    self.db_manager.update_websocket_status(symbol, 'kline', timeframe, False)
            except Exception as e:
                print(f"Error closing WebSocket {key}: {e}")
                self.db_manager.log_event('ERROR', f"Error closing WebSocket {key}: {e}", 'BinanceClient')

        self.active_websockets.clear()

    def cleanup(self):
        """
           Повністю очищає всі ресурси, закриває всі WebSocket-з'єднання
           та розриває з'єднання з базою даних.

           Викликається зазвичай при завершенні роботи клієнта.
           """
        self.close_all_websockets()
        self.db_manager.disconnect()
        print("Cleaned up all resources")

    # ===== Сервісні функції для перепідключення та моніторингу =====

    def check_and_handle_reconnections(self):
        """
           Перевіряє, чи встановлено прапорець необхідності перепідключення,
           і, якщо так, виконує перевірку WebSocket-з'єднань та їх перепідключення.

           Після виконання скидає прапорець `self.reconnect_required`.
           """
        if self.reconnect_required:
            print("Reconnection flag detected, checking WebSocket connections...")
            self.db_manager.log_event('INFO', "Reconnection flag detected, checking WebSocket connections...",
                                      'BinanceClient')
            self.check_websocket_connections()
            self.reconnect_required = False

    # ===== Збереження історичних даних в базу даних =====

    def _prepare_historical_params(self, symbol, start_date=None, end_date=None, timeframe=None):
        """
            Підготовка параметрів для збору історичних даних про криптовалюту.

            Параметри:
                symbol (str): Символ торгової пари (наприклад, 'BTCUSDT').
                start_date (str або tuple): Початкова дата у форматі 'YYYY-MM-DD' або кортеж (symbol, date).
                end_date (str): Кінцева дата у форматі 'YYYY-MM-DD' (необов'язково).
                timeframe (List[str]): Список таймфреймів для збору даних (наприклад, ['1m', '1h']).

            Повертає:
                dict: Словник з параметрами:
                    - 'timeframe': Список таймфреймів
                    - 'start_ts': Початковий timestamp у мс
                    - 'end_ts': Кінцевий timestamp у мс
                    - 'start_date': Початкова дата у вигляді рядка
                    - 'end_date': Кінцева дата або "сьогодні"
            """
        # Словник дат початку для символів
        symbol_start_dates = {
            'BTCUSDT': '2017-08-01',
            'ETHUSDT': '2017-08-01',
            'SOLUSDT': '2020-08-27'
        }

        # Встановлення значень за замовчуванням
        if not timeframe:
            timeframe = ['1m', '1h', '1d']

        # Обробка start_date
        if isinstance(start_date, tuple) and len(start_date) == 2:
            # Якщо start_date є кортежем (symbol, date)
            symbol_specific_date = start_date[1]
            start_date = symbol_specific_date
        elif not start_date:
            # Використовуємо дату з словника, якщо вона існує, інакше використовуємо 30 днів тому
            start_date = symbol_start_dates.get(symbol,
                        (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))

        # Конвертуємо дати в timestamp
        try:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            if end_date:
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            else:
                end_ts = int(datetime.now().timestamp() * 1000)
        except ValueError as e:
            self.db_manager.log_event('ERROR', f"Помилка формату дати: {e}", 'BinanceClient')
            print(f"Помилка формату дати: {e}")
            raise ValueError(f"Неправильний формат дати: {e}")

        # Виділяємо базовий актив та котирувальний актив
        # Це буде працювати для форматів XXXUSDT, але потребує доопрацювання для інших пар
        is_valid, base_asset = self._validate_symbol(symbol)
        quote_asset = symbol[len(base_asset):]

        # Зберігаємо інформацію про криптовалюту
        if is_valid:
            self.db_manager.insert_cryptocurrency(symbol, base_asset, quote_asset)
        else:
            raise ValueError(f"Непідтримуваний символ: {symbol}")

        # Повідомлення про початок збору даних
        self.db_manager.log_event('INFO',
                                  f"Починаємо зберігання історичних даних для {symbol} з інтервалами {timeframe}",
                                  'BinanceClient')
        print(f"Починаємо зберігання історичних даних для {symbol} з інтервалами {timeframe}")

        return {
            'timeframe': timeframe,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'start_date': start_date,
            'end_date': end_date if end_date else 'сьогодні'
        }
    def _calculate_window_size(self, timeframe):
        """
        Розрахунок оптимального розміру вікна для запиту в залежності від інтервалу
        """
        if timeframe == '1m':
            # Для хвилинних даних обмежуємо запит до 12 годин (720 хвилин)
            return 1000 * 60 * 60 * 12  # 12 годин в мілісекундах
        elif timeframe == '1h':
            # Для годинних даних обмежуємо запит до 10 днів (240 годин)
            return 1000 * 60 * 60 * 24 * 10  # 10 днів в мілісекундах
        else:
            # Для інших інтервалів використовуємо 100 днів
            return 1000 * 60 * 60 * 24 * 100  # 100 днів в мілісекундах

    def _process_and_save_klines_batch(self, symbol, timeframe, df):
        """
           Обробляє пакет свічок (Kline) і зберігає їх у відповідну таблицю бази даних.

           Параметри:
               symbol (str): Назва торгової пари, наприклад 'BTCUSDT'.
               timeframe (str): Інтервал часу свічок (наприклад '1m', '1h').
               df (pd.DataFrame): DataFrame, який містить Kline-дані.

           Повертає:
               int: Кількість успішно збережених свічок.

           Обробка включає:
               - Конвертацію даних до відповідного формату.
               - Валідацію символу.
               - Вставку кожного запису в таблицю {symbol}_klines.
               - Логування помилок при невдалому збереженні.
           """
        saved_count = 0

        if df.empty:
            return 0

        for _, row in df.iterrows():
            try:
                # Підготовка даних свічки
                kline_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'open_time': row['open_time'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'close_time': row['close_time'],
                    'quote_asset_volume': float(row['quote_asset_volume']),
                    'number_of_trades': int(row['number_of_trades']),
                    'taker_buy_base_volume': float(row['taker_buy_base_asset_volume']),
                    'taker_buy_quote_volume': float(row['taker_buy_quote_asset_volume']),
                    'is_closed': True
                }

                # Отримуємо базову валюту з символа
                is_valid, crypto_symbol = self._validate_symbol(symbol)

                if not is_valid:
                    continue

                # Вставка в базу даних в таблицю {symbol}_klines
                table_name = f"{symbol}_klines"  # Формат таблиці: BTCUSDT_klines, ETHUSDT_klines, тощо
                result = self.db_manager.insert_kline(crypto_symbol, kline_data, table_name)

                if result:
                    saved_count += 1

            except Exception as e:
                error_msg = f"Помилка збереження свічки {symbol} ({timeframe}) в БД: {e}"
                self.db_manager.log_event('ERROR', error_msg, 'BinanceClient')
                print(error_msg)

        return saved_count

    def _fetch_and_save_historical_interval(self, symbol, timeframe, start_ts, end_ts):
        """
    Збирає та зберігає історичні Kline-дані для заданого таймфрейму в межах зазначеного інтервалу часу.

    Параметри:
        symbol (str): Назва торгової пари (наприклад 'BTCUSDT').
        timeframe (str): Таймфрейм (наприклад '1m', '1h', '1d').
        start_ts (int): Початковий час у мілісекундах (timestamp).
        end_ts (int): Кінцевий час у мілісекундах (timestamp).

    Повертає:
        int: Загальна кількість збережених свічок.

    Логіка роботи:
        - Інтервал часу розбивається на вікна розміру, що відповідає таймфрейму.
        - Для кожного вікна виконується API-запит до Binance.
        - Усі отримані свічки зберігаються в базу даних.
        - Уникається дублювання за рахунок точного оновлення меж запитів.
        - У разі помилок логуються повідомлення та виконується повторна спроба.
    """
        saved_candles_count = 0
        current_start = start_ts
        window_size = self._calculate_window_size(timeframe)

        self.db_manager.log_event('INFO',
                                  f"Збір даних для {symbol} з інтервалом {timeframe} (часове вікно: {datetime.fromtimestamp(start_ts / 1000)} - {datetime.fromtimestamp(end_ts / 1000)})",
                                  'BinanceClient')

        # Проходимо через весь часовий діапазон
        while current_start < end_ts:
            try:
                # Визначаємо кінець поточного вікна
                current_end = min(current_start + window_size, end_ts)

                # Виводимо діапазон дат, з якого збираємо дані
                start_date_str = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m-%d %H:%M:%S')
                end_date_str = datetime.fromtimestamp(current_end / 1000).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Отримання даних {symbol} ({timeframe}) від {start_date_str} до {end_date_str}")

                # Отримуємо дані свічок
                df = self.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=current_start,
                    end_time=current_end,
                    limit=1000,  # Максимальна кількість свічок за один запит
                    use_cache=False  # Не використовуємо кеш для історичних даних
                )

                if df.empty:
                    log_msg = f"Дані відсутні для {symbol} ({timeframe}) від {start_date_str} до {end_date_str}"
                    self.db_manager.log_event('WARNING', log_msg, 'BinanceClient')
                    print(log_msg)

                    # Переходимо до наступного вікна
                    current_start = current_end + 1
                    continue

                # Обробка та збереження отриманих даних
                batch_saved = self._process_and_save_klines_batch(symbol, timeframe, df)
                saved_candles_count += batch_saved

                # Виводимо прогрес
                if batch_saved > 0:
                    print(f"Збережено {batch_saved} свічок (всього: {saved_candles_count}) для {symbol} ({timeframe})")

                # Оновлюємо початок вікна на основі останньої отриманої свічки
                if len(df) > 0:
                    # Додаємо 1 мс до часу закриття останньої свічки, щоб уникнути дублювання
                    current_start = int(df.iloc[-1]['close_time'].timestamp() * 1000) + 1
                else:
                    # Якщо пустий датафрейм, то переходимо до наступного вікна
                    current_start = current_end + 1

                # Затримка, щоб уникнути перевищення ліміту запитів API
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                error_msg = f"Помилка API запиту для {symbol} ({timeframe}): {e}"
                self.db_manager.log_event('ERROR', error_msg, 'BinanceClient')
                print(error_msg)
                time.sleep(5)  # Довша затримка у випадку помилки API
                current_start = current_end + 1  # Пропускаємо поточне вікно
                continue

            except Exception as e:
                error_msg = f"Неочікувана помилка при обробці даних {symbol} ({timeframe}): {e}"
                self.db_manager.log_event('ERROR', error_msg, 'BinanceClient')
                print(error_msg)
                current_start = current_end + 1  # Пропускаємо поточне вікно
                continue

        # Виводимо підсумок для інтервалу
        summary_msg = f"Завершено збір даних для {symbol} ({timeframe}). Збережено {saved_candles_count} свічок."
        self.db_manager.log_event('INFO', summary_msg, 'BinanceClient')
        print(summary_msg)

        return saved_candles_count

    def _summarize_historical_results(self, symbol, results):
        """
    Підводить підсумки після збору історичних Kline-даних по різних таймфреймах.

    Параметри:
        symbol (str): Назва торгової пари (наприклад 'BTCUSDT').
        results (dict): Словник, де ключ — таймфрейм, а значення — кількість збережених свічок.

    Дії:
        - Виводить у консоль кількість збережених свічок по кожному таймфрейму.
        - Формує загальне зведення по всіх таймфреймах.
        - Логує загальне зведення у систему логування.
    """
        total_candles = sum(results.values())

        # Для кожного інтервалу виводимо кількість збережених свічок
        for timeframe, count in results.items():
            print(f"{symbol} ({timeframe}): збережено {count} свічок")

        # Загальний підсумок
        summary_msg = f"Загалом для {symbol} збережено {total_candles} свічок по всіх інтервалах."
        self.db_manager.log_event('INFO', summary_msg, 'BinanceClient')
        print(summary_msg)

    def save_historical_data_to_db(self, symbol, start_date=None, end_date=None, timeframe=None):
        """
           Збирає історичні Kline-дані з Binance та зберігає їх у базу даних.

           Параметри:
               symbol (str): Назва торгової пари (наприклад 'ETHUSDT').
               start_date (str або tuple, optional): Дата початку у форматі 'YYYY-MM-DD' або кортеж (symbol, date).
               end_date (str, optional): Дата завершення у форматі 'YYYY-MM-DD'. Якщо не задано — використовується поточна дата.
               timeframe (list або str, optional): Один або кілька таймфреймів (наприклад ['1h', '1d']).

           Повертає:
               dict: Словник з кількістю збережених свічок для кожного таймфрейму.

           Особливості:
               - Автоматично виконує підготовку параметрів.
               - Під час помилок проводить логування і повертає порожній словник.
               - В кінці викликає підсумкову функцію для виводу результатів.
           """
        try:
            # Підготовка параметрів
            params = self._prepare_historical_params(symbol, start_date, end_date, timeframe)

            # Словник для збереження результатів
            results = {}

            # Збираємо дані для кожного інтервалу
            for timeframe in params['timeframe']:
                results[timeframe] = self._fetch_and_save_historical_interval(
                    symbol, timeframe, params['start_ts'], params['end_ts']
                )

            # Підведення підсумків
            self._summarize_historical_results(symbol, results)
            return results

        except Exception as e:
            error_msg = f"Помилка при зборі історичних даних для {symbol}: {e}"
            self.db_manager.log_event('ERROR', error_msg, 'BinanceClient')
            print(error_msg)
            return {}

def handle_kline_message(ws, message):
    """
       Обробляє повідомлення WebSocket про нову свічку (Kline) від Binance.

       Параметри:
           ws: Об'єкт WebSocket (не використовується безпосередньо, але потрібен для сумісності з інтерфейсом).
           message (str): Повідомлення у форматі JSON, що містить інформацію про нову свічку.

       Дії:
           - Розпарсює JSON-повідомлення.
           - Виводить у консоль основні характеристики нової свічки:
               - символ
               - інтервал
               - час відкриття
               - ціни (open, high, low, close)
               - обсяг
               - статус закриття свічки
       """
    data = json.loads(message)
    candle = data['k']

    print(f"Нова свічка {candle['s']} {candle['i']}:")
    print(f"Час відкриття: {datetime.fromtimestamp(candle['t'] / 1000)}")
    print(f"Ціна відкриття: {candle['o']}")
    print(f"Найвища ціна: {candle['h']}")
    print(f"Найнижча ціна: {candle['l']}")
    print(f"Поточна ціна: {candle['c']}")
    print(f"Обсяг: {candle['v']}")
    print(f"Ціна закриття: {candle['x']}")
    print("-----")


def main():
    client = BinanceClient()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Початкові дати для кожного символу
    symbol_start_dates = {
        'BTCUSDT': '2017-08-01',
        'ETHUSDT': '2017-08-01',
        'SOLUSDT': '2020-08-27'
    }

    # Колекція отриманих через WebSocket свічок для подальшого аналізу
    received_candles = {symbol: {'1m': [], '1h': [], '1d': []} for symbol in symbols}

    # Функція-обробник WebSocket повідомлень із збереженням свічок
    def handle_kline_message_with_collection(ws, message):
        data = json.loads(message)
        candle = data['k']

        # Базовий вивід інформації
        print(f"Нова свічка {candle['s']} {candle['i']}:")
        print(f"Час відкриття: {datetime.fromtimestamp(candle['t'] / 1000)}")
        print(f"Ціна відкриття: {candle['o']}")
        print(f"Найвища ціна: {candle['h']}")
        print(f"Найнижча ціна: {candle['l']}")
        print(f"Поточна ціна: {candle['c']}")
        print(f"Обсяг: {candle['v']}")
        print(f"Ціна закриття: {candle['x']}")
        print("-----")

        # Збереження свічки для подальшої обробки
        symbol = candle['s']
        interval = candle['i']

        # Створюємо структуру даних для збереження свічки
        candle_data = {
            'symbol': symbol,
            'interval': interval,
            'open_time': datetime.fromtimestamp(candle['t'] / 1000),
            'open': float(candle['o']),
            'high': float(candle['h']),
            'low': float(candle['l']),
            'close': float(candle['c']),
            'volume': float(candle['v']),
            'close_time': datetime.fromtimestamp(candle['T'] / 1000),
            'quote_asset_volume': float(candle['q']),
            'number_of_trades': int(candle['n']),
            'taker_buy_base_volume': float(candle['V']),
            'taker_buy_quote_volume': float(candle['Q']),
            'is_closed': bool(candle['x'])
        }

        # Додаємо свічку до колекції
        if symbol in received_candles and interval in received_candles[symbol]:
            received_candles[symbol][interval].append(candle_data)

            # Якщо свічка закрита, виводимо повідомлення і можна додатково обробити
            if candle_data['is_closed']:
                print(f"Закрита свічка {symbol} {interval}: {candle_data['open_time']} - {candle_data['close_time']}")
                # Тут можна додати код для аналізу закритих свічок

    try:
        # Збір історичних даних
        print("========== ЗБІР ІСТОРИЧНИХ ДАНИХ ==========")

        # Поточна дата і час як кінцева точка
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Збираємо дані до поточного часу: {end_date}")

        # Для кожного символу — збір даних з початкової дати до поточного часу
        for symbol in symbols:
            print(f"\nЗбираємо історичні дані для {symbol}...")

            # Отримати початкову дату для конкретного символу або дефолт
            start_date = (symbol, '2025-05-18')

            # Збір даних для всіх інтервалів: 1 хв, 1 год, 1 день до поточного моменту
            results = client.save_historical_data_to_db(
                symbol=symbol,
                start_date=start_date,
                end_date=datetime.now().strftime('%Y-%m-%d'),  # Поточна дата
                timeframe=['1m', '1h', '1d']
            )

            print(f"Результат для {symbol}: {results}")

            # Перевірка наявності останніх свічок (включаючи поточну незакриту)
            print(f"\nПеревірка останніх свічок для {symbol}...")

            for timeframe in ['1m', '1h', '1d']:
                # Отримуємо найсвіжішу свічку
                latest_kline = client.get_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1,
                    use_cache=False
                )

                if not latest_kline.empty:
                    last_candle_time = latest_kline.iloc[0]['close_time']
                    current_time = pd.Timestamp.now()
                    time_diff = current_time - last_candle_time

                    print(f"{symbol} {timeframe} остання свічка: {last_candle_time}")
                    print(f"Різниця з поточним часом: {time_diff}")

                    # Якщо різниця більша за очікувану для даного таймфрейму, виводимо попередження
                    expected_diff = pd.Timedelta(minutes=1)
                    if timeframe == '1h':
                        expected_diff = pd.Timedelta(hours=1)
                    elif timeframe == '1d':
                        expected_diff = pd.Timedelta(days=1)

                    if time_diff > expected_diff:
                        print(f"УВАГА: Дані {symbol} {timeframe} можуть бути неповними!")

        print("\n========== ОТРИМАННЯ АКТУАЛЬНИХ ДАНИХ ==========")
        # Отримання поточних цін
        for symbol in symbols:
            try:
                price = client.get_ticker_price(symbol=symbol)
                if price:
                    print(f"Поточна ціна для {symbol}: {price['price']}")
                else:
                    print(f"Не вдалося отримати ціну для {symbol}")
            except Exception as e:
                print(f"Помилка під час отримання ціни для {symbol}: {e}")

        print("\n========== ПІДКЛЮЧЕННЯ ДО WEBSOCKET ==========")
        print("Підключення до WebSocket для отримання даних у реальному часі...")

        active_sockets = []
        # Запуск WebSocket для кожного символу (для 1m, 1h, 1d)
        for symbol in symbols:
            try:
                # Використовуємо оновлений обробник, який також зберігає свічки
                socket_1m = client.start_kline_socket(symbol, "1m", handle_kline_message_with_collection, save_to_db=True)
                socket_1h = client.start_kline_socket(symbol, "1h", handle_kline_message_with_collection, save_to_db=True)
                socket_1d = client.start_kline_socket(symbol, "1d", handle_kline_message_with_collection, save_to_db=True)

                for socket in [socket_1m, socket_1h, socket_1d]:
                    if socket:
                        active_sockets.append(socket)

                print(f"WebSocket для {symbol} успішно підключено (1m, 1h, 1d)")

            except Exception as e:
                print(f"Помилка створення WebSocket для {symbol}: {e}")

        if not active_sockets:
            print("Не вдалося створити жодне WebSocket з'єднання!")
            return

        print("Очікування ринкових даних у реальному часі. Натисніть Ctrl+C для виходу.")

        # Основний цикл з обробкою отриманих даних
        collection_interval = 60  # Секунди між обробкою зібраних даних
        last_collection_time = time.time()

        while True:
            time.sleep(5)
            client.check_and_handle_reconnections()

            # Періодична обробка зібраних свічок
            current_time = time.time()
            if current_time - last_collection_time >= collection_interval:
                print("\n===== АНАЛІЗ ЗІБРАНИХ СВІЧОК =====")

                # Для кожного символу аналізуємо зібрані свічки
                for symbol in symbols:
                    for timeframe in ['1m', '1h', '1d']:
                        collected = received_candles[symbol][timeframe]
                        closed_candles = [c for c in collected if c['is_closed']]

                        if closed_candles:
                            print(f"Зібрано {len(closed_candles)} закритих свічок {symbol} {timeframe}")

                            # Створюємо DataFrame для аналізу
                            if closed_candles:
                                df = pd.DataFrame(closed_candles)

                                # Базовий аналіз: мінімальна, максимальна ціна та середній об'єм
                                if not df.empty:
                                    min_price = df['low'].min()
                                    max_price = df['high'].max()
                                    avg_volume = df['volume'].mean()

                                    print(f"  Мінімальна ціна: {min_price}")
                                    print(f"  Максимальна ціна: {max_price}")
                                    print(f"  Середній об'єм: {avg_volume}")

                                    # Очищаємо оброблені закриті свічки
                                    received_candles[symbol][timeframe] = [c for c in collected if not c['is_closed']]

                # Скидаємо таймер для наступної обробки
                last_collection_time = current_time

                # Додатково перевіряємо наявність пропусків в історичних даних
                # і заповнюємо їх при необхідності
                for symbol in symbols:
                    for timeframe in ['1m', '1h', '1d']:
                        try:
                            # Отримуємо останню свічку з бази даних
                            last_db_candle = client.db_manager.get_klines(symbol, timeframe)

                            if last_db_candle:
                                last_time = last_db_candle['close_time']
                                current_time = datetime.now()

                                # Визначаємо різницю часу
                                time_diff = (current_time - last_time).total_seconds()

                                # Визначаємо очікувану різницю залежно від таймфрейму
                                expected_diff = 60  # 1 хвилина в секундах
                                if timeframe == '1h':
                                    expected_diff = 3600  # 1 година в секундах
                                elif timeframe == '1d':
                                    expected_diff = 86400  # 1 день в секундах

                                # Якщо є значний розрив, заповнюємо відсутні дані
                                if time_diff > expected_diff * 2:  # Множимо на 2 для запасу
                                    print(f"Виявлено пропуск даних для {symbol} {timeframe}. Заповнення...")

                                    # Конвертуємо в формат часу для API
                                    start_ts = int(last_time.timestamp() * 1000)
                                    end_ts = int(current_time.timestamp() * 1000)

                                    # Заповнюємо пропущені дані
                                    missing_candles = client._fetch_and_save_historical_interval(
                                        symbol, timeframe, start_ts, end_ts
                                    )

                                    print(f"Заповнено {missing_candles} пропущених свічок для {symbol} {timeframe}")

                        except Exception as e:
                            print(f"Помилка при перевірці пропусків даних для {symbol} {timeframe}: {e}")

    except KeyboardInterrupt:
        print("Завершення роботи...")
    except Exception as e:
        print(f"Критична помилка: {e}")
    finally:
        # Аналіз зібраних свічок перед виходом
        print("\n===== ФІНАЛЬНИЙ АНАЛІЗ ЗІБРАНИХ СВІЧОК =====")
        for symbol in symbols:
            for timeframe in ['1m', '1h', '1d']:
                collected = received_candles[symbol][timeframe]
                if collected:
                    print(f"Всього зібрано {len(collected)} свічок {symbol} {timeframe}")

        client.cleanup()


if __name__ == "__main__":
    main()