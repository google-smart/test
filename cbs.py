Traceback (most recent call last):
  File "D:\Program Files (x86)\Anaconda3\lib\site-packages\pandas\core\indexes\base.py", line 3629, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas\_libs\index.pyx", line 136, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 163, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 5198, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 5206, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'date'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "E:\资料\BaiduSyncdisk\我的助理\虚拟货币\BTC.py", line 35, in <module>
    print(df['date'].head())
  File "D:\Program Files (x86)\Anaconda3\lib\site-packages\pandas\core\frame.py", line 3505, in __getitem__
    indexer = self.columns.get_loc(key)
  File "D:\Program Files (x86)\Anaconda3\lib\site-packages\pandas\core\indexes\base.py", line 3631, in get_loc
    raise KeyError(key) from err
KeyError: 'date'
