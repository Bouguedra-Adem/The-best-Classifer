{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import itertools\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.ticker import NullFormatter\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.ticker as ticker\n",
    "from sklearn import preprocessing\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### About dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:\n",
    "\n",
    "| Field          | Description                                                                           |\n",
    "|----------------|---------------------------------------------------------------------------------------|\n",
    "| Loan_status    | Whether a loan is paid off on in collection                                           |\n",
    "| Principal      | Basic principal loan amount at the                                                    |\n",
    "| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |\n",
    "| Effective_date | When the loan got originated and took effects                                         |\n",
    "| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |\n",
    "| Age            | Age of applicant                                                                      |\n",
    "| Education      | Education of applicant                                                                |\n",
    "| Gender         | The gender of applicant                                                               |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets download the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'wget' n’est pas reconnu en tant que commande interne\n",
      "ou externe, un programme exécutable ou un fichier de commandes.\n"
     ]
    }
   ],
   "source": [
    "!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Data From CSV File  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>9/22/2016</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>10/8/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30       9/8/2016   \n",
       "1           2             2     PAIDOFF       1000     30       9/8/2016   \n",
       "2           3             3     PAIDOFF       1000     15       9/8/2016   \n",
       "3           4             4     PAIDOFF       1000     30       9/9/2016   \n",
       "4           6             6     PAIDOFF       1000     30       9/9/2016   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2016   45  High School or Below    male  \n",
       "1  10/7/2016   33              Bechalor  female  \n",
       "2  9/22/2016   27               college    male  \n",
       "3  10/8/2016   28               college  female  \n",
       "4  10/8/2016   29               college    male  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(346, 10)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Convert to date time object "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0 2016-10-07   45  High School or Below    male  \n",
       "1 2016-10-07   33              Bechalor  female  \n",
       "2 2016-09-22   27               college    male  \n",
       "3 2016-10-08   28               college  female  \n",
       "4 2016-10-08   29               college    male  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['due_date'] = pd.to_datetime(df['due_date'])\n",
    "df['effective_date'] = pd.to_datetime(df['effective_date'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Data visualization and pre-processing\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Let’s see how many of each class is in our data set "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PAIDOFF       260\n",
       "COLLECTION     86\n",
       "Name: loan_status, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['loan_status'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "260 people have paid off the loan on time while 86 have gone into collection \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets plot some columns to underestand data better:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Solving environment: done\n",
      "\n",
      "## Package Plan ##\n",
      "\n",
      "  environment location: /opt/conda/envs/Python36\n",
      "\n",
      "  added / updated specs: \n",
      "    - seaborn\n",
      "\n",
      "\n",
      "The following packages will be downloaded:\n",
      "\n",
      "    package                    |            build\n",
      "    ---------------------------|-----------------\n",
      "    ca-certificates-2020.1.1   |                0         132 KB  anaconda\n",
      "    seaborn-0.10.0             |             py_0         161 KB  anaconda\n",
      "    openssl-1.1.1              |       h7b6447c_0         5.0 MB  anaconda\n",
      "    certifi-2019.11.28         |           py36_0         156 KB  anaconda\n",
      "    ------------------------------------------------------------\n",
      "                                           Total:         5.5 MB\n",
      "\n",
      "The following packages will be UPDATED:\n",
      "\n",
      "    ca-certificates: 2019.11.27-0       --> 2020.1.1-0        anaconda\n",
      "    certifi:         2019.11.28-py36_0  --> 2019.11.28-py36_0 anaconda\n",
      "    openssl:         1.1.1d-h7b6447c_3  --> 1.1.1-h7b6447c_0  anaconda\n",
      "    seaborn:         0.9.0-pyh91ea838_1 --> 0.10.0-py_0       anaconda\n",
      "\n",
      "\n",
      "Downloading and Extracting Packages\n",
      "ca-certificates-2020 | 132 KB    | ##################################### | 100% \n",
      "seaborn-0.10.0       | 161 KB    | ##################################### | 100% \n",
      "openssl-1.1.1        | 5.0 MB    | ##################################### | 100% \n",
      "certifi-2019.11.28   | 156 KB    | ##################################### | 100% \n",
      "Preparing transaction: done\n",
      "Verifying transaction: done\n",
      "Executing transaction: done\n"
     ]
    }
   ],
   "source": [
    "# notice: installing seaborn might takes a few minutes\n",
    "!conda install -c anaconda seaborn -y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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IIcccgjnnHMODzzwQMt2ffr0aXkkxwUXXMD8+fP53Oc+11L3nDlzGDFiBMOGDePwww9nzpw5LesuvPBCBg4cyMaNGwFYu3YtNTU1HfqdtJcft2FmZW3lO29z1HWPFK2+Z7+Vb+zsrUUEZ555JhMnTmwZtbyhoYHVq1dz4YUXMnXqVMaNG8f69es566yzuPPOO1tGiuiIL3zhCy2jnDfbsGEDp5xyCt/97nc59dRTAXjiiSeorq5uGX5p9OjRTJkypWW8vvnz57fs/+KLL3LFFVcwb948amtreeONNzjxxBM54IADGDFiBJA8W2r69OlccsklHY65I3wFZWZWZE888QRVVVVcfPHFLWV1dXUsX76cY489lnHjxgGwyy67cPvtt3PTTTcV7dj33XcfRx99dEtyAvjsZz/b8kDEtkyZMoVrr72W2tpaAGpra7nmmmu49dZbW7aZPHkyt912G5s3by5a3Pk4QZmZFdnSpUs/8UgMyP+ojCFDhvDhhx/ywQcfdPg4ud12dXV1fPTRR60eu73a8ziPwYMHc9xxx/HjH/+408dpD3fxmZl1k20fi5GrtfLtydfFV6h8MeYru/baaznttNM45ZRTinr8XL6CMjMrskMPPZRFixblLd92XMXXX3+dPn360Ldv3y49dkf23zbGfI/zOPDAA6mrq+PBBx/s9LHa4gRlZlZkY8aMYePGjdx9990tZc8//zxDhw5lwYIF/PrXvwaSBxtefvnlXHnllUU79he/+EV+85vf8POf/7yl7JFHHuGll15q1/5XXHEF3/72t2lsbASgsbGRG2+8ka9//euf2PYb3/gGU6ZMKUrc+biLz8zK2oCB+7XrzruO1NcWScyePZvJkydz00030bt3b2pqavje977HQw89xFe/+lUuvfRStmzZwvnnn7/VreUzZszY6rbuZ55JngE7YsQIdtghuaY455xzGDFiBA888MBWz5O68847OeaYY5g7dy6TJ09m8uTJVFVVMWLECL7//e+36/XV1dVx8803c+qpp7Jp0yaqqqq45ZZbWp4QnOvQQw9l1KhR7X50fUe163EbXc2PE7Du4MdtVAY/biM7Cn3cRptdfJKmS3pP0tKcslslvSppiaTZknZPy2skfSSpIZ1+2N5AzMzMcrXnM6gZwLbXx/OAwyJiBLAcuCZn3e8joi6dLsbMzKwT2kxQEfEk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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "\n",
    "bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'Principal', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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ZMCAAaxbtw6AlStXMnjw4Hb9TQrhufgKMKi6mg+XLSt2GGbWQUs//ojDb3g6tfpe/tfxbW4TEZxxxhlMmDChZdbyhoYGPv30Uy688ELuu+8+xo0bx5o1azjzzDO59957W2aKaI/vfOc7LbOcN1u7di0nnXQSd9xxB6eccgoAs2bNoqqqqmX6pdGjR3P77be3zNc3e/bslve//vrrXHXVVcycOZPa2lo++OADTjjhBPbZZx+GDx8O5O4t9eCDD3LppZe2O+ZCOUEVIK17yfguuGY9x6xZs6ioqOCSSy5pKaurq+OBBx5g1KhRjBs3DoAddtiBu+++m9GjR3coQeXzyCOPcOSRR7YkJ4Djjjuu4PfffvvtXH/99dTW1gJQW1vLddddx2233cavf/1rACZNmsSdd97J9773vVRizsddfGZmXWDhwoWb3RID8t8qY99992X16tV88cUX7d5P6267uro6vvrqqy3uu1CF3M5j4MCBHH300S0Jqyv4DMrMrBtteluM1rZUvjX5uvg6K1+M+cquv/56Tj31VE466aRU99/MZ1BmZl3goIMOYt68eXnL586du1HZ+++/T58+fejbt2+X7rs97980xny389hvv/2oq6vj8ccf7/C+tsYJysysC4wZM4Z169bxy1/+sqXs1VdfZciQIcyZM4dnn30WyN3Y8IorruDqq69Obd/f/e53+fOf/8zvf//7lrKnn36aN954o6D3X3XVVfz0pz+lsbERgMbGRm666SauvPLKzbb90Y9+xO23355K3JtyF5+Zlb3qAXsXNPKuPfW1RRLTpk1j0qRJ3HzzzVRWVjJ48GDuuusunnzySS6//HImTpzIhg0bOP/88zcaWj5lypSNhnW/9NJLAAwfPpxttsmdV5xzzjkMHz6cxx57bKP7Sd17770cddRRzJgxg0mTJjFp0iQqKioYPnw4P/vZzwo6vrq6Om655RZOOeUUvv76ayoqKrj11ltb7hDc2kEHHcTIkSMLvnV9exR0u4201NfXx6anjaUgrdtk1HzS5NttlL8O3W6jtVJtJ1ni221kR3fcbsPMzKxbOUGZmVkmOUGZWVlyF3jxdfZv4ARlZmWnsrKSVatWOUkVUUSwatUqKisrO1yHR/GZWdmpqamhqamJFStWFDuUHq2yspKamo4PDHOCKlHb0bFfneczsH9/lixdmkpdZllQUVHRMo+clS4nqBK1DlIdsm5mljUFX4OS1EvSa5JmJOu1kl6W9K6kxyRt23VhmplZT9OeQRI/ABa3Wr8FuDMihgCfARelGZiZmfVsBSUoSTXAScCvknUBY4Ankk0eAk7vigDNzKxnKvQM6i7gauCbZH034POIWJ+sNwED8r1R0sWS5kqa6xE1Zvm5nZhtrs0EJelkYHlEtJ67Pd/wsbw/OIiIyRFRHxH1VVVVHQzTrLy5nZhtrpBRfKOAUyWdCFQCO5E7o9pFUu/kLKoG+KTrwjQzs56mzTOoiLguImoiYjBwLvB8RJwHzALOSjabADzZZVGamVmP05mpjq4BfijpPXLXpB5IJyQzM7N2/lA3ImYDs5Pn7wOHpR+SmZmZJ4s1M7OMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMcoIyM7NMajNBSaqU9Iqk1yUtkvSTpLxW0suS3pX0mKRtuz5cMzPrKQo5g1oHjImIbwF1wHhJRwC3AHdGxBDgM+CirgvTzMx6mjYTVOSsTlYrkiWAMcATSflDwOldEqGZmfVIBV2DktRLUgOwHJgJ/BX4PCLWJ5s0AQO28N6LJc2VNHfFihVpxGxWdtxOzDZXUIKKiA0RUQfUAIcBB+bbbAvvnRwR9RFRX1VV1fFIzcqY24nZ5to1ii8iPgdmA0cAu0jqnbxUA3ySbmhmZtaTFTKKr0rSLsnz7YHjgcXALOCsZLMJwJNdFaSZmfU8vdvehGrgIUm9yCW0xyNihqQ3gUcl/RvwGvBAF8ZpZmY9TJsJKiIWACPylL9P7nqUmZlZ6jyThJmZZZITlJmZZZITlJmZZZITlJmZZVLZJqhB1dVISmUxM7PuV8gw85L04bJlNO1Vk0pdNZ80pVKPmZkVrmzPoMzMrLQ5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSY5QZmZWSa1maAk7S1plqTFkhZJ+kFS3k/STEnvJo+7dn24ZmbWUxRyBrUeuDIiDgSOACZKGgZcCzwXEUOA55J1MzOzVLSZoCJiaUTMT55/CSwGBgCnAQ8lmz0EnN5VQZqZWc/TrmtQkgYDI4CXgT0jYinkkhiwxxbec7GkuZLmrlixonPRmpUptxOzzRWcoCT1AX4HTIqILwp9X0RMjoj6iKivqqrqSIxmZc/txGxzBSUoSRXkktPDEfHvSfGnkqqT16uB5V0TopmZ9USFjOIT8ACwOCLuaPXSU8CE5PkE4Mn0w7PusB20edv7QpZB1dXFPhQzKyOF3PJ9FHA+8IakhqTseuBm4HFJFwEfAmd3TYjW1dYBTXvVdLqemk+aOh+MmVmizQQVEXMAbeHlsemGk03qVZHKf77qvW1q/4mrV0Uq9ZiZZVUhZ1A9Xmz4msNveLrT9bz8r+NTqae5LjOzcuapjszMLJOcoMzMLJOcoMzMLJOcoMzMLJOcoMzMLJOcoMzMLJOcoMzMLJOcoMzMLJOcoMzMLJPKdiaJtKYnMjOz4ijbBJXW9ETgaYXMzIrBXXxmZpZJTlBmZpZJTlBmZpZJZXsNqtylOQjE95ayrBlUXc2Hy5Z1up7tt+nFV99sSCEiGNi/P0uWLk2lLiuME1SJ8iAQK2cfLluW2l2e06inuS7rXm128Ul6UNJySQtblfWTNFPSu8njrl0bppmZ9TSFXIOaAmz6Ffta4LmIGAI8l6xbD7cdICmVZVB1dbEPx8yKrM0uvoj4k6TBmxSfBoxOnj8EzAauSTEuK0HrwN0pZpaajo7i2zMilgIkj3tsaUNJF0uaK2nuihUrOrg7s/JWDu1kUHV1amfQZtANgyQiYjIwGaC+vj66en9mpagc2klaAxvAZ9CW09EzqE8lVQMkj8vTC8nMzKzjCeopYELyfALwZDrhmJmZ5RQyzHwq8BfgAElNki4CbgZOkPQucEKybmZmlppCRvH9yxZeGptyLGZmZi0yNRefRwGZmVmzTE115FFAZmbWLFMJyoojrYlnPemsmaXJCcpSm3jWk86aWZoydQ3KzMysmROUmZllkhOUmZllkhOUmZllkhOUZZLvLdU9/NtDyzKP4rNM8r2luod/e2hZ5gRlqUnr91TNdZlZz+YEZalJ6/dU4N9UmZmvQZmZWUb5DMoyKc3uwm16VaRyEX9g//4sWbo0hYjKU6pdvL239fRbBRhUXc2Hy5alUlcWP99OUJZJaXcXpjEQwIMAti7tv5mn32pbuQ9ycRefmZllUqbOoNLsIjAzs9KWqQTlUWBmZtasUwlK0njgZ0Av4FcRcXMqUZmlqBzvd5XmxXErTFqDbQC26V3BN+u/TqWuctbhBCWpF3APcALQBLwq6amIeDOt4MzSUI73u0rr4ri71Av3jQfudLvODJI4DHgvIt6PiH8AjwKnpROWmZn1dIqIjr1ROgsYHxH/I1k/Hzg8Ii7bZLuLgYuT1QOAtzsebovdgZUp1JMFPpZs6uixrIyIdp9qdVE7Af9NsqqnH0tB7aQz16DydcZulu0iYjIwuRP72XzH0tyIqE+zzmLxsWRTdx9LV7QT8N8kq3wshelMF18TsHer9Rrgk86FY2ZmltOZBPUqMERSraRtgXOBp9IJy8zMeroOd/FFxHpJlwHPkBtm/mBELEotsq1LvSukiHws2VQux1IuxwE+lqzqsmPp8CAJMzOzruS5+MzMLJOcoMzMLJMyn6Ak7S1plqTFkhZJ+kFS3k/STEnvJo+7FjvWtkiqlPSKpNeTY/lJUl4r6eXkWB5LBp1knqRekl6TNCNZL8njAJDUKOkNSQ2S5iZlJfMZczvJtnJpK93dTjKfoID1wJURcSBwBDBR0jDgWuC5iBgCPJesZ906YExEfAuoA8ZLOgK4BbgzOZbPgIuKGGN7/ABY3Gq9VI+j2XERUdfqNx2l9BlzO8m2cmor3ddOIqKkFuBJcvP/vQ1UJ2XVwNvFjq2dx7EDMB84nNyvsHsn5UcCzxQ7vgLir0k+jGOAGeR+uF1yx9HqeBqB3TcpK9nPmNtJdpZyaivd3U5K4QyqhaTBwAjgZWDPiFgKkDzuUbzICpec6jcAy4GZwF+BzyNifbJJEzCgWPG1w13A1cA3yfpulOZxNAvgj5LmJdMOQel+xgbjdpIl5dRWurWdZOp+UFsjqQ/wO2BSRHyR1rT33S0iNgB1knYBpgEH5tuse6NqH0knA8sjYp6k0c3FeTbN9HFsYlREfCJpD2CmpLeKHVBHuJ1kSxm2lW5tJyWRoCRVkGt0D0fEvyfFn0qqjoilkqrJfdMqGRHxuaTZ5K4X7CKpd/KNqhSmjBoFnCrpRKAS2Inct8RSO44WEfFJ8rhc0jRys/WX1GfM7SSTyqqtdHc7yXwXn3JfAR8AFkfEHa1eegqYkDyfQK7PPdMkVSXfCJG0PXA8uQuns4Czks0yfywRcV1E1ETEYHJTXD0fEedRYsfRTNKOkvo2PwfGAQspoc+Y20k2lVNbKUo7KfZFtwIuyh1N7vR3AdCQLCeS68d9Dng3eexX7FgLOJbhwGvJsSwEbkjK9wFeAd4DfgtsV+xY23FMo4EZpXwcSdyvJ8si4EdJecl8xtxOsr+UelspRjvxVEdmZpZJme/iMzOznskJyszMMskJyszMMskJyszMMskJyszMMskJyszMMskJyszMMskJqsRJmp5M3LioefJGSRdJekfSbEm/lHR3Ul4l6XeSXk2WUcWN3qx7uJ2UJv9Qt8RJ6hcRf0umhHkV+C/Ai8BI4EvgeeD1iLhM0iPAvRExR9JAclP855uE06ysuJ2UppKYLNa26gpJZyTP9wbOB/5vRPwNQNJvgf2T148HhrWa4XonSX0j4svuDNisCNxOSpATVAlLpu8/HjgyItYksz6/Tf5bE0CuS/fIiPiqeyI0Kz63k9Lla1ClbWfgs6TRDSV3S4IdgG9L2lVSb+DMVtv/EbiseUVSXbdGa1YcbiclygmqtD0N9Ja0ALgReAn4GLiJ3N1UnwXeBP6ebH8FUC9pgaQ3gUu6P2Szbud2UqI8SKIMSeoTEauTb4bTgAcjYlqx4zLLEreT7PMZVHn6saQGcvfS+QCYXuR4zLLI7STjfAZlZmaZ5DMoMzPLJCcoMzPLJCcoMzPLJCcoMzPLJCcoMzPLpP8PlTlGZbaTvVAAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "bins = np.linspace(df.age.min(), df.age.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'age', bins=bins, ec=\"k\")\n",
    "\n",
    "g.axes[-1].legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Pre-processing:  Feature selection/extraction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Lets look at the day of the week people get the loan "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['dayofweek'] = df['effective_date'].dt.dayofweek\n",
    "bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)\n",
    "g = sns.FacetGrid(df, col=\"Gender\", hue=\"loan_status\", palette=\"Set1\", col_wrap=2)\n",
    "g.map(plt.hist, 'dayofweek', bins=bins, ec=\"k\")\n",
    "g.axes[-1].legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>female</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below    male          3        0  \n",
       "1 2016-10-07   33              Bechalor  female          3        0  \n",
       "2 2016-09-22   27               college    male          3        0  \n",
       "3 2016-10-08   28               college  female          4        1  \n",
       "4 2016-10-08   29               college    male          4        1  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Convert Categorical features to numerical values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets look at gender:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender  loan_status\n",
       "female  PAIDOFF        0.865385\n",
       "        COLLECTION     0.134615\n",
       "male    PAIDOFF        0.731293\n",
       "        COLLECTION     0.268707\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "86 % of female pay there loans while only 73 % of males pay there loan\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets convert male to 0 and female to 1:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>dayofweek</th>\n",
       "      <th>weekend</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>45</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-10-07</td>\n",
       "      <td>33</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>2016-09-08</td>\n",
       "      <td>2016-09-22</td>\n",
       "      <td>27</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>28</td>\n",
       "      <td>college</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>6</td>\n",
       "      <td>6</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>2016-09-09</td>\n",
       "      <td>2016-10-08</td>\n",
       "      <td>29</td>\n",
       "      <td>college</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           0             0     PAIDOFF       1000     30     2016-09-08   \n",
       "1           2             2     PAIDOFF       1000     30     2016-09-08   \n",
       "2           3             3     PAIDOFF       1000     15     2016-09-08   \n",
       "3           4             4     PAIDOFF       1000     30     2016-09-09   \n",
       "4           6             6     PAIDOFF       1000     30     2016-09-09   \n",
       "\n",
       "    due_date  age             education  Gender  dayofweek  weekend  \n",
       "0 2016-10-07   45  High School or Below       0          3        0  \n",
       "1 2016-10-07   33              Bechalor       1          3        0  \n",
       "2 2016-09-22   27               college       0          3        0  \n",
       "3 2016-10-08   28               college       1          4        1  \n",
       "4 2016-10-08   29               college       0          4        1  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## One Hot Encoding  \n",
    "#### How about education?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "education             loan_status\n",
       "Bechalor              PAIDOFF        0.750000\n",
       "                      COLLECTION     0.250000\n",
       "High School or Below  PAIDOFF        0.741722\n",
       "                      COLLECTION     0.258278\n",
       "Master or Above       COLLECTION     0.500000\n",
       "                      PAIDOFF        0.500000\n",
       "college               PAIDOFF        0.765101\n",
       "                      COLLECTION     0.234899\n",
       "Name: loan_status, dtype: float64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(['education'])['loan_status'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Feature befor One Hot Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>education</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>High School or Below</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>Bechalor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>college</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender             education\n",
       "0       1000     30   45       0  High School or Below\n",
       "1       1000     30   33       1              Bechalor\n",
       "2       1000     15   27       0               college\n",
       "3       1000     30   28       1               college\n",
       "4       1000     30   29       0               college"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[['Principal','terms','age','Gender','education']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "#### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Feature = df[['Principal','terms','age','Gender','weekend']]\n",
    "Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)\n",
    "Feature.drop(['Master or Above'], axis = 1,inplace=True)\n",
    "Feature.head()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Feature selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Lets defind feature sets, X:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>weekend</th>\n",
       "      <th>Bechalor</th>\n",
       "      <th>High School or Below</th>\n",
       "      <th>college</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>45</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1000</td>\n",
       "      <td>15</td>\n",
       "      <td>27</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Principal  terms  age  Gender  weekend  Bechalor  High School or Below  \\\n",
       "0       1000     30   45       0        0         0                     1   \n",
       "1       1000     30   33       1        0         1                     0   \n",
       "2       1000     15   27       0        0         0                     0   \n",
       "3       1000     30   28       1        1         0                     0   \n",
       "4       1000     30   29       0        1         0                     0   \n",
       "\n",
       "   college  \n",
       "0        0  \n",
       "1        0  \n",
       "2        1  \n",
       "3        1  \n",
       "4        1  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = Feature\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "What are our lables?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['loan_status'].values\n",
    "y[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Normalize Data "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Data Standardization give data zero mean and unit variance (technically should be done after train test split )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Anaconda\\lib\\site-packages\\sklearn\\preprocessing\\data.py:645: DataConversionWarning: Data with input dtype uint8, int64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "C:\\Anaconda\\lib\\site-packages\\ipykernel_launcher.py:1: DataConversionWarning: Data with input dtype uint8, int64 were all converted to float64 by StandardScaler.\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 0.51578458,  0.92071769,  2.33152555, -0.42056004, -1.20577805,\n",
       "        -0.38170062,  1.13639374, -0.86968108],\n",
       "       [ 0.51578458,  0.92071769,  0.34170148,  2.37778177, -1.20577805,\n",
       "         2.61985426, -0.87997669, -0.86968108],\n",
       "       [ 0.51578458, -0.95911111, -0.65321055, -0.42056004, -1.20577805,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.48739188,  2.37778177,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679],\n",
       "       [ 0.51578458,  0.92071769, -0.3215732 , -0.42056004,  0.82934003,\n",
       "        -0.38170062, -0.87997669,  1.14984679]])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X= preprocessing.StandardScaler().fit(X).transform(X)\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "# Classification "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model\n",
    "You should use the following algorithm:\n",
    "- K Nearest Neighbor(KNN)\n",
    "- Decision Tree\n",
    "- Support Vector Machine\n",
    "- Logistic Regression\n",
    "\n",
    "\n",
    "\n",
    "__ Notice:__ \n",
    "- You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.\n",
    "- You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.\n",
    "- You should include the code of the algorithm in the following cells."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# K Nearest Neighbor(KNN)\n",
    "Notice: You should find the best k to build the model with the best accuracy.  \n",
    "**warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 233,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection  import  train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 234,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "best score is : 0.7857142857142857 best K is:  7\n"
     ]
    }
   ],
   "source": [
    "k_knn=15\n",
    "accuracy_knn=[]\n",
    "\n",
    "#training model and find best K \n",
    "for k in range (1,k_knn) :\n",
    "    knn=KNeighborsClassifier(n_neighbors=k)\n",
    "    knn=knn.fit(x_train,y_train)\n",
    "    accuracy_knn.append(metrics.accuracy_score(y_test,knn.predict(x_test)))\n",
    "    \n",
    "#print best accuracy and best K\n",
    "accuracy_knn=np.array(accuracy_knn)\n",
    "print('best score is :',accuracy_knn.max() ,'best K is: ' ,accuracy_knn.argmax()+1)\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>lets do some visualisation to undrestand well</h3> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 259,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(range(1,accuracy_knn.shape[0]+1),accuracy_knn,'g',markerfacecolor='blue', markersize=8, marker='o' ,)\n",
    "plt.xlabel('K' ,fontsize=20)\n",
    "plt.ylabel('Accuracy' ,fontsize=20)\n",
    "plt.title('Find Best K' ,fontsize=20)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3>Bulid model with K=7 and Evaluation with different metrics </h3>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 262,
   "metadata": {},
   "outputs": [],
   "source": [
    " knn=KNeighborsClassifier(n_neighbors=7)\n",
    " knn=knn.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "decision=tree.DecisionTreeClassifier(min_samples_split=10)\n",
    "decision=decision.fit(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION', 'PAIDOFF',\n",
       "       'COLLECTION', 'COLLECTION', 'COLLECTION', 'COLLECTION',\n",
       "       'COLLECTION', 'COLLECTION', 'PAIDOFF', 'COLLECTION', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF'], dtype=object)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "decision.predict(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import svm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "SVM=svm.SVC()\n",
    "SVM. probability=True\n",
    "SVM=SVM.fit(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "SVM.predict(X)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Anaconda\\lib\\site-packages\\sklearn\\linear_model\\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.\n",
      "  FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "LogisticReg = LogisticRegression(random_state=0)\n",
    "LogisticReg=LogisticReg.fit(X,y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'COLLECTION',\n",
       "       'PAIDOFF', 'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF',\n",
       "       'COLLECTION', 'PAIDOFF', 'PAIDOFF', 'PAIDOFF'], dtype=object)"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LogisticReg.predict(X)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Evaluation using Test set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import jaccard_similarity_score\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import log_loss\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, download and load the test set:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "'wget' n’est pas reconnu en tant que commande interne\n",
      "ou externe, un programme exécutable ou un fichier de commandes.\n"
     ]
    }
   ],
   "source": [
    "!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Load Test set for evaluation "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Unnamed: 0.1</th>\n",
       "      <th>loan_status</th>\n",
       "      <th>Principal</th>\n",
       "      <th>terms</th>\n",
       "      <th>effective_date</th>\n",
       "      <th>due_date</th>\n",
       "      <th>age</th>\n",
       "      <th>education</th>\n",
       "      <th>Gender</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/8/2016</td>\n",
       "      <td>10/7/2016</td>\n",
       "      <td>50</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>300</td>\n",
       "      <td>7</td>\n",
       "      <td>9/9/2016</td>\n",
       "      <td>9/15/2016</td>\n",
       "      <td>35</td>\n",
       "      <td>Master or Above</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>21</td>\n",
       "      <td>21</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2016</td>\n",
       "      <td>10/9/2016</td>\n",
       "      <td>43</td>\n",
       "      <td>High School or Below</td>\n",
       "      <td>female</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>24</td>\n",
       "      <td>24</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>1000</td>\n",
       "      <td>30</td>\n",
       "      <td>9/10/2016</td>\n",
       "      <td>10/9/2016</td>\n",
       "      <td>26</td>\n",
       "      <td>college</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>35</td>\n",
       "      <td>35</td>\n",
       "      <td>PAIDOFF</td>\n",
       "      <td>800</td>\n",
       "      <td>15</td>\n",
       "      <td>9/11/2016</td>\n",
       "      <td>9/25/2016</td>\n",
       "      <td>29</td>\n",
       "      <td>Bechalor</td>\n",
       "      <td>male</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0  Unnamed: 0.1 loan_status  Principal  terms effective_date  \\\n",
       "0           1             1     PAIDOFF       1000     30       9/8/2016   \n",
       "1           5             5     PAIDOFF        300      7       9/9/2016   \n",
       "2          21            21     PAIDOFF       1000     30      9/10/2016   \n",
       "3          24            24     PAIDOFF       1000     30      9/10/2016   \n",
       "4          35            35     PAIDOFF        800     15      9/11/2016   \n",
       "\n",
       "    due_date  age             education  Gender  \n",
       "0  10/7/2016   50              Bechalor  female  \n",
       "1  9/15/2016   35       Master or Above    male  \n",
       "2  10/9/2016   43  High School or Below  female  \n",
       "3  10/9/2016   26               college    male  \n",
       "4  9/25/2016   29              Bechalor    male  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')\n",
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Anaconda\\lib\\site-packages\\sklearn\\preprocessing\\data.py:645: DataConversionWarning: Data with input dtype uint8, int64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "C:\\Anaconda\\lib\\site-packages\\ipykernel_launcher.py:13: DataConversionWarning: Data with input dtype uint8, int64 were all converted to float64 by StandardScaler.\n",
      "  del sys.path[0]\n"
     ]
    }
   ],
   "source": [
    "test_df['due_date'] = pd.to_datetime(test_df['due_date'])\n",
    "test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])\n",
    "test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek\n",
    "test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)\n",
    "test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)\n",
    "\n",
    "Feature_test = test_df[['Principal','terms','age','Gender','weekend']]\n",
    "Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)\n",
    "Feature_test.drop(['Master or Above'], axis = 1,inplace=True)\n",
    "\n",
    "y_test = test_df['loan_status'].values\n",
    "X_test = Feature_test\n",
    "X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "Jaccard=[]\n",
    "F1score=[]\n",
    "LogLoss=[]\n",
    "\n",
    "y_pred1=k_grd.predict(X_test)\n",
    "y_pred2=decision.predict(X_test)\n",
    "y_pred3=SVM.predict(X_test)\n",
    "y_pred4=LogisticReg.predict(X_test)\n",
    "\n",
    "y_pred1_proba=k_grd.predict_proba(X_test)\n",
    "y_pred2_proba=decision.predict_proba(X_test)\n",
    "y_pred3_proba=SVM.predict_proba(X_test)\n",
    "y_pred4_proba=LogisticReg.predict_proba(X_test)\n",
    "\n",
    "Jaccard=[jaccard_similarity_score(y_test,y_pred1),jaccard_similarity_score(y_test,y_pred2),jaccard_similarity_score(y_test,y_pred3),jaccard_similarity_score(y_test,y_pred4)]\n",
    "F1score=[f1_score(y_test,y_pred1,average='weighted'),f1_score(y_test,y_pred2,average='weighted'),f1_score(y_test,y_pred3,average='weighted'),f1_score(y_test,y_pred4,average='weighted')]\n",
    "LogLoss=[log_loss(y_test,y_pred1_proba),log_loss(y_test,y_pred2_proba),log_loss(y_test,y_pred3_proba),log_loss(y_test,y_pred4_proba)]\n",
    "        \n",
    "\n",
    "\n",
    "d = {'Algorithm': ['KNN', 'Decision Tree','SVM','LogisticRegression'], 'Jaccard':np.array(4)  }\n",
    "df_evaluation = pd.DataFrame(data=d)\n",
    "df_evaluation.set_index(df_evaluation['Algorithm'],inplace=True )\n",
    "df_evaluation=df_evaluation.drop(columns=['Algorithm'])\n",
    "\n",
    "df_evaluation['Jaccard']=Jaccard\n",
    "df_evaluation['F1-score']=F1score\n",
    "df_evaluation['LogLoss']=LogLoss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Jaccard</th>\n",
       "      <th>F1-score</th>\n",
       "      <th>LogLoss</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Algorithm</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>KNN</th>\n",
       "      <td>0.722222</td>\n",
       "      <td>0.710576</td>\n",
       "      <td>1.716432</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Decision Tree</th>\n",
       "      <td>0.666667</td>\n",
       "      <td>0.657911</td>\n",
       "      <td>2.329016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SVM</th>\n",
       "      <td>0.722222</td>\n",
       "      <td>0.621266</td>\n",
       "      <td>0.554366</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LogisticRegression</th>\n",
       "      <td>0.759259</td>\n",
       "      <td>0.671764</td>\n",
       "      <td>0.479193</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Jaccard  F1-score   LogLoss\n",
       "Algorithm                                       \n",
       "KNN                 0.722222  0.710576  1.716432\n",
       "Decision Tree       0.666667  0.657911  2.329016\n",
       "SVM                 0.722222  0.621266  0.554366\n",
       "LogisticRegression  0.759259  0.671764  0.479193"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Report\n",
    "You should be able to report the accuracy of the built model using different evaluation metrics:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "| Algorithm          | Jaccard | F1-score | LogLoss |\n",
    "|--------------------|---------|----------|---------|\n",
    "| KNN                | ?       | ?        | NA      |\n",
    "| Decision Tree      | ?       | ?        | NA      |\n",
    "| SVM                | ?       | ?        | NA      |\n",
    "| LogisticRegression | ?       | ?        | ?       |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<h2>Want to learn more?</h2>\n",
    "\n",
    "IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href=\"http://cocl.us/ML0101EN-SPSSModeler\">SPSS Modeler</a>\n",
    "\n",
    "Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href=\"https://cocl.us/ML0101EN_DSX\">Watson Studio</a>\n",
    "\n",
    "<h3>Thanks for completing this lesson!</h3>\n",
    "\n",
    "<h4>Author:  <a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a></h4>\n",
    "<p><a href=\"https://ca.linkedin.com/in/saeedaghabozorgi\">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>\n",
    "\n",
    "<hr>\n",
    "\n",
    "<p>Copyright &copy; 2018 <a href=\"https://cocl.us/DX0108EN_CC\">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href=\"https://bigdatauniversity.com/mit-license/\">MIT License</a>.</p>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
