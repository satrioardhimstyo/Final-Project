---
layout: post
title:  "Analysis FIFA 19 Players Dataset"
date:   2020-10-13 11:12:20 +0700
categories: jekyll update
---
##  Introduction
Pada analisis ini menggunakan dataset berasal dari [karangadiya](https://www.kaggle.com/karangadiya/fifa19) yang mana berisikan dataset mengenai pemain dari game Fifa 19 dengan atribut dari data.csv :

Age, Nationality, Overall, Potential, Club, Value, Wage, Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight, LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB, Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, dan Release Clause.

Proses analisis ini meliputi Data Cleaning, EDA dan Modeling Machine learning. Pada Modeling Machine Learning memakai metode Scaling Power Transformer. Serta menggunakan algoritma LinearRegression, RandomForest, SVR, dan Deep Learning menggunakan opt Rmsprop.

### - Importing Data
Berikut ini adalah 5 data pertama dari dataset [karangadiya](https://www.kaggle.com/karangadiya/fifa19):

[![2ZUWDg.th.png](https://iili.io/2ZUWDg.th.png)](https://freeimage.host/i/2ZUWDg) [![2ZUVOF.th.png](https://iili.io/2ZUVOF.th.png)](https://freeimage.host/i/2ZUVOF) [![2ZUMR1.th.png](https://iili.io/2ZUMR1.th.png)](https://freeimage.host/i/2ZUMR1) 

[![2ZUGHP.th.png](https://iili.io/2ZUGHP.th.png)](https://freeimage.host/i/2ZUGHP) [![2ZUhxa.th.png](https://iili.io/2ZUhxa.th.png)](https://freeimage.host/i/2ZUhxa) [![2ZUjWJ.th.png](https://iili.io/2ZUjWJ.th.png)](https://freeimage.host/i/2ZUjWJ)

Foto diatas adalah foto data mentah yg belum dilakukan cleaning yang mana masih terdapat missing value.
Langkah berikutnya adalah cleaning data.

### - Data Cleaning
Step pertama yaitu mengganti isi dari atribut yang memiliki missing value yang awalnya bertanda "?" menjadi nan, dan dilakukan pengecekan berapa banyak missing value yg ada pada datase ini .
{% highlight ruby %}
data.replace("?", np.nan, inplace=True)
pd.set_option('display.max_rows', 100)
data.isnull().sum().sort_values()
{% endhighlight %}

[![2Z4VO7.th.png](https://iili.io/2Z4VO7.th.png)](https://freeimage.host/i/2Z4VO7)

Setelah didapatkan missing value dari dataset ini ternyata missing value dengan jumlah 48, selanjutnya akan di cek apakah semuanya memiliki kesamaan atau tidak.

{% highlight ruby %}
missing_height = data[data['Height'].isnull()].index.tolist()
missing_weight = data[data['Weight'].isnull()].index.tolist()
if missing_height == missing_weight:
    print('They are same')
else:
    print('They are different')
{% endhighlight %}
[![2Z6Iqb.th.png](https://iili.io/2Z6Iqb.th.png)](https://freeimage.host/i/2Z6Iqb)

Bisa diasumsikan mereka memliki nilai sama karena hasinya seperti diatas, maka dapat disimpulkan data yg lain pun sama. Step berikutnya akan dilakukan juga drop ke feature yg sekiranya tidak akan dipakai untuk kedepanya.

{% highlight ruby %}
data.drop(data.index[missing_height],inplace =True)
data.drop(['Unnamed: 0','Joined','ID','Photo','Flag','Club Logo','Loaned From','Release Clause','Jersey Number','Club'],axis=1,inplace=True)
data.drop(['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB'], axis = 1, inplace=True)
{% endhighlight %}

Berikutnya melakukan pengisian nilai nan pada masing masinng feature yg masih terdapat missing value dengan nilai mean dan median dari feature tersebut.

{% highlight ruby %}
data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace=True)
data['Volleys'].fillna(data['Volleys'].mean(), inplace=True)
data['Dribbling'].fillna(data['Dribbling'].mean(), inplace=True)
data['Curve'].fillna(data['Curve'].mean(), inplace=True)
data['FKAccuracy'].fillna(data['FKAccuracy'].mean(), inplace=True)
data['LongPassing'].fillna(data['LongPassing'].mean(), inplace=True)
data['BallControl'].fillna(data['BallControl'].mean(), inplace=True)
data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace=True)
data['Finishing'].fillna(data['Finishing'].mean(), inplace=True)
data['Crossing'].fillna(data['Crossing'].mean(), inplace=True)
data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace=True)
{% endhighlight %}

Dilakukan juga pengisian nilai pada feature yang memiliki missing value dengan nilai yang di isi kan manual.

{% highlight ruby %}
data['Weak Foot'].fillna(3, inplace=True)
data['Position'].fillna('CAM', inplace=True)
data['Preferred Foot'].fillna('Right', inplace=True)
data['International Reputation'].fillna(1, inplace=True)
data['Contract Valid Until'].fillna(2020, inplace=True)
{% endhighlight %}

[![2ZPwcx.th.png](https://iili.io/2ZPwcx.th.png)](https://freeimage.host/i/2ZPwcx) [![2ZPZNI.th.png](https://iili.io/2ZPZNI.th.png)](https://freeimage.host/i/2ZPZNI)

Sudah tidak terdapat missing value dari masing masing feature yang ada pada dataset fifa19 ini.

### - Exploratory Data Analysis

40 besar Negara dengan pemain terbanyak di fifa 19
{% highlight ruby %}
data_nat = data['Nationality'].value_counts().reset_index()
data_nat.columns = ['Nationality', 'count']
data_nat = data_nat.sort_values('count')
fig = px.bar(
    data_nat.tail(40), 
    x='count',
    y="Nationality", 
    orientation='h', 
    title='Top 40 Nationality by number of players', 
    height=800, 
    width=800
)
fig.show()
{% endhighlight %}

[![2ZpS7n.md.png](https://iili.io/2ZpS7n.md.png)](https://freeimage.host/i/2ZpS7n)

Urutan pemain dengan Overall tertinggi hingga terendah

{% highlight ruby %}
overall = data.sort_values('Overall', ascending=False)[['Name', 'Age', 'Overall', 'Nationality']]
overall
{% endhighlight %}

[![2ZpPLl.png](https://iili.io/2ZpPLl.png)](https://freeimage.host/id)

40 Negara dengan Overall pemain terbaik

{% highlight ruby %}
data_ov = data.groupby('Nationality')['Overall'].max().reset_index().sort_values('Overall', ascending=True).tail(40)

fig = px.bar(
    data_ov, 
    x="Overall", 
    y="Nationality", 
    orientation='h',
    width=800,
    height=800
)

fig.show()
{% endhighlight %}

[![2Zpt29.md.png](https://iili.io/2Zpt29.md.png)](https://freeimage.host/i/2Zpt29)

Persebaran pemain berdasarkan umur dan overall pemain

{% highlight ruby %}
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='Age', y='Overall', data=data, palette='winter')
ax.set_xlabel(xlabel='Age')
ax.set_ylabel(ylabel='Overall')
ax.set_title(label='Overall Player Base Age', fontsize=15)
plt.show()
{% endhighlight %}

[![2ZyM4S.md.png](https://iili.io/2ZyM4S.md.png)](https://freeimage.host/i/2ZyM4S)

Korelasi dari masing masing feature yang ada

{% highlight ruby %}
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="YlGnBu")
{% endhighlight %}

[![2Zy48F.md.png](https://iili.io/2Zy48F.md.png)](https://freeimage.host/i/2Zy48F)


## -Machine Learning

Sebelum melakukan modeling, dilakukan cleaning dengan mendrop beberapa feature kembali dan mendefinisi beberapa feature

{% highlight ruby %}
data.drop(['Special','Body Type','Weight','Height','Contract Valid Until','Wage','Value','Name','Position','Work Rate'], axis = 1, inplace=True)
{% endhighlight %}

{% highlight ruby %}
def face_to_num(data):
    if (data['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
#Turn Preferred Foot into a binary indicator variable
def right_footed(data):
    if (data['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

#Get a count of Nationalities in the Dataset, make of list of those with over 250 Players (our Major Nations)
nat_counts = data.Nationality.value_counts()
nat_list = nat_counts[nat_counts > 250].index.tolist()

#Replace Nationality with a binary indicator variable for 'Major Nation'
def major_nation(data):
    if (data.Nationality in nat_list):
        return 1
    else:
        return 0

#Create a copy of the original dataframe to avoid indexing errors
data1 = data.copy()

#Apply changes to dataset to create new column
data1['Real_Face'] = data1.apply(face_to_num, axis=1)
data1['Right_Foot'] = data1.apply(right_footed, axis=1)
data1['Major_Nation'] = data1.apply(major_nation,axis = 1)

#Drop original columns used
data1 = data1.drop(['Preferred Foot','Real Face','Nationality'], axis = 1)
{% endhighlight %}

### Linear Regression

Algortima pertama yang digunakan untuk machine learning adalah linear regression dengan menggunakan scaler Power Transformer. Dan dilakukan pendefinisain label dan feature kemudian dilakukan data split pada data train dan test nya 

{% highlight ruby %}
label = data1['Overall']
feature = data1.drop(['Overall'], axis = 1)

feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size = 0.25, random_state = 10)
{% endhighlight %}

Kemudian digunakan pipeline agar mempermudah untuk meyimpan data yg sudah di scaling, agar mempermudah untuk deploy ke web service.

{% highlight ruby %}
pipeline = Pipeline([('scaler', PowerTransformer()), ('model', LinearRegression())])
pipeline.fit(feature_train, label_train)
{% endhighlight %}

{% highlight ruby %}
predict_linreg_train = pipeline.predict(feature_train)
mse_train = mean_squared_error(label_train, predict_linreg_train)
r2_train = r2_score(label_train, predict_linreg_train)
print('r2 score :', r2_train)
print('RMSE score:', np.sqrt(mse_train))
print('')
predict_linreg = pipeline.predict(feature_test)
mse = mean_squared_error(label_test, predict_linreg)
mae = mean_absolute_error(label_test, predict_linreg)
r2 = r2_score(label_test, predict_linreg)
print('MSE :', mse)
print('MAE :', mae)
print('r2 score :', r2)
print('RMSE score  :', np.sqrt(mse))
{% endhighlight %}

[![2bwOPV.th.png](https://iili.io/2bwOPV.th.png)](https://freeimage.host/i/2bwOPV)

### SVR
Sama seperti proses yang dilakukan di Regresi hanya ada tambahan untuk memakai parameter dengan menggunakan Randomized Search.

{% highlight ruby %}
svr = SVR()
C = [0.0001,0.001, 0.01, 1.0, 10.0, 100.0, 1000.0]
gamma = [10.0,1.0, 0.1, 0.01, 0.001,000.1]
{% endhighlight %}

{% highlight ruby %}
svr_parameter = {'model__C' : [0.0001,0.001, 0.01,1.0, 10.0, 100.0, 1000.0], 'model__gamma' : [10.0,1.0, 0.1, 0.01, 0.001]}
print(svr_parameter)
{% endhighlight %}

{% highlight ruby %}
pipeline_svr = Pipeline([('scaler', PowerTransformer()), ('model', SVR(kernel='rbf'))])
svr_random = RandomizedSearchCV(pipeline_svr, svr_parameter, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
svr_random.fit(feature_train, label_train) 
print("")

svr_random.best_params_
{% endhighlight %}
paramater yang digunakan pada algoritma ada nilai dari C, gamma dan kernel. Dapat dilihat pula bahwa parameter terbaik yang menghasilkan nilai maksimal yaitu dengan nilai gamma 100 dan C 0.01 denga kernel yang digunkan adalah rbf.

{% highlight ruby %}
predict_svr_train = pipeline_svr.predict(feature_train)
r2_train = r2_score(label_train, predict_svr_train)
mse_train = mean_squared_error(label_train, predict_svr_train)
print('r2 score :', r2_train)
print('RMSE score :', np.sqrt(mse_train))
print("")

predict_svr_test = pipeline_svr.predict(feature_test)
mse = mean_squared_error(label_test, predict_svr_test)
mae = mean_absolute_error(label_test, predict_svr_test)
r2 = r2_score(label_test, predict_svr_test)
print('MSE  :', mse)
print('MAE  :', mae)
print('r2 score :', r2)
print('RMSE score :', np.sqrt(mse))
{% endhighlight %}

[![2b4rn1.th.png](https://iili.io/2b4rn1.th.png)](https://freeimage.host/i/2b4rn1)

### Random Forest 

{% highlight ruby %}
randomf = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [1,3,5,7,9] 
min_samples_leaf = [1,3,5,7,9]
bootstrap = [True, False]

random_parameter = {'model__n_estimators' : n_estimators,
               'model__max_features' : max_features, 
               'model__max_depth' : max_depth,
               'model__min_samples_split' : min_samples_split,
               'model__min_samples_leaf' : min_samples_leaf,
               'model__bootstrap' : bootstrap}

print(random_parameter)
{% endhighlight %}

{% highlight ruby %}
pipeline_randomf = Pipeline([('scaler', PowerTransformer()), ('model', RandomForestRegressor())])

randomf_random = RandomizedSearchCV(pipeline_randomf, random_parameter, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
randomf_random.fit(feature_train, label_train)

randomf_random.best_params_
{% endhighlight %}

Parameter terbaik yang menghasilkan nilai maksimum pada algoritma random forest yaitu model__bootstrap: False model__max_depth: 70 , model__max_features: sqrt, model__min_samples_leaf: 1, model__min_samples_split: 7,model__n_estimators: 1000.

{% highlight ruby %}
pipeline_randomf = Pipeline([('scaler', PowerTransformer()), ('model', RandomForestRegressor(bootstrap=False, max_depth=70, max_features='sqrt', min_samples_leaf=1, min_samples_split=7, n_estimators=1000))])
pipeline_randomf.fit(feature_train, label_train)
{% endhighlight %}
{% highlight ruby %}
predict_randomf_train = pipeline_randomf.predict(feature_train)
r2_train = r2_score(label_train, predict_randomf_train)
mse_train = mean_squared_error(label_train, predict_randomf_train)
print('r2 score :', r2_train)
print('RMSE score :', np.sqrt(mse_train))
print(" ")

predict_randomf = pipeline_randomf.predict(feature_test)
mse = mean_squared_error(label_test, predict_randomf)
mae = mean_absolute_error(label_test, predict_randomf)
r2 = r2_score(label_test, predict_randomf)
print('MSE :', mse)
print('MAE :', mae)
print('r2 score:', r2)
print('RMSE score:', np.sqrt(mse))
{% endhighlight %}

[![3cVAgt.th.png](https://iili.io/3cVAgt.th.png)](https://freeimage.host/i/3cVAgt)

Diatas adalah hasil dari learning menggunakan algoritma random forest.

## - Deep Learning
Algoritma selanjutnya akan digunakn deep leraning karena hasilnya didapatkan jika hanyak mengandalkan machine learning ternyata masih belum cukup optimal hasilnya. Disini akan menggunakan 2 laye dengan layer pertama memakai 42 neuron dan layer kedua 13 neuron dengna optimizer RMSprop dengan learning rate sebesar 0.001 dan momentum 0.9.
{% highlight ruby %}
feature_deep = data1.drop(['Overall'], axis=1)
label_deep = data1['Overall']
{% endhighlight %}
{% highlight ruby %}
robust_feature = RobustScaler()
robust_label = RobustScaler()
scaled_feature = robust_feature.fit(feature_deep)
scaled_label = robust_label.fit(data1['Overall'].values.reshape(-1, 1))
{% endhighlight %}

Perbedaan yang tampak pada proses deep learning ini feature dan labelnya akan dilakukan scaling secara terpisah menggunakan robust scaler.
{% highlight ruby %}
save_scaled_feature = 'scaled_feature.pkl' 
pickle.dump(scaled_feature, open(save_scaled_feature, 'wb')) 

save_scaled_label = 'scaled_label.pkl' 
pickle.dump(scaled_label, open(save_scaled_label, 'wb'))
{% endhighlight %}

Agar mempermudah untuk proses deploy web servicenya maka feature dan label yang sudah dilakukan scaling maka akan di save menggunakan pickle.
{% highlight ruby %}
scaled_feature_deep = scaled_feature.transform(feature_deep)
scaled_label_deep = scaled_label.transform(data1['Overall'].values.reshape(-1, 1)).flatten()
{% endhighlight %}
{% highlight ruby %}
feature_deep_train, feature_deep_test, label_deep_train, label_deep_test = train_test_split(scaled_feature_deep, scaled_label_deep, test_size=0.25, random_state=10)
{% endhighlight %}
{% highlight ruby %}
model = Sequential()
model.add(Dense(42, input_dim=42, kernel_initializer='uniform', activation='relu'))
model.add(Dense(13, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform'))

opt = RMSprop(lr=0.001, momentum=0.9)

model.summary()

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['CosineSimilarity'])
{% endhighlight %}  
[![3cXrba.th.png](https://iili.io/3cXrba.th.png)](https://freeimage.host/i/3cXrba)

Diatas adalah merupakan model summary untuk model deep learning.

{% highlight ruby %}
filepath="weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
{% endhighlight %} 
Hasil terbaik nanti akan disave kedalam folder bernama weights dan nanti akan dilakukan checkpoint pula pada val loss dengan nilai yang paling rendah.
{% highlight ruby %}
history = model.fit(feature_deep_train, label_deep_train, validation_data=(feature_deep_test, label_deep_test), epochs=100, callbacks=callbacks_list,batch_size=32, verbose=0)
{% endhighlight %}
{% highlight ruby %}
predict_deep = model.predict(feature_deep_test)
predict_deep = predict_deep.flatten()

mse = mean_squared_error(label_deep_test, predict_deep)
mae = mean_absolute_error(label_deep_test, predict_deep)
r2 = r2_score(label_deep_test, predict_deep)
print("MSE          :", mse)
print("MAE          :", mae)
print("r2 score     :", r2)
print('RMSE score   :', np.sqrt(mse))
{% endhighlight %}
[![3chM1p.th.png](https://iili.io/3chM1p.th.png)](https://freeimage.host/i/3chM1p)

[![3chVrN.th.png](https://iili.io/3chVrN.th.png)](https://freeimage.host/i/3chVrN)

## - Deploy Web Service 
Langkah terakhir akan dilakukan deployment model ke web service memakai insomnnia dengan mempredeksi data dari label 16133, yang mana sebelumnya sudah di lihat bahwa memiliki overall dengan nilai 58.
{% highlight ruby %}
loaded_model = load_model('/content/weights_best.h5')
scaler_feature = pickle.load(open(save_scaled_feature, 'rb'))
scaler_label = pickle.load(open(save_scaled_label, 'rb'))
{% endhighlight %}
{% highlight ruby %}
data_testing = [[17,76,1.0,3.0,3.0,61.0,48.0,43.0,55.0,52.0,65.0,54.0,44.0,54.0,64.0,67.0,65.0,69.0,46.0,69.0,51.0,62.0,49.0,51.0,41.0,39.0,22.0,44.0,42.0,49.0,48.0,33.0,26.0,24.0,7.0,12.0,9.0,11.0,6.0,0,1,1]]

scaling_data_testing = scaler_feature.transform(test_data)
predict_model = loaded_model.predict(scaling_data_testing)
invers_predict = scaler_label.inverse_transform(predict_model)
print('Overall This Player {}'.format(invers_predict[0]))
{% endhighlight ruby %}

Hasil yang didapatkan yaitu 56.59502 sedikit meleset dari data aktualnya.
{% highlight ruby %}
app = Flask(__name__) 

@app.route("/home")
def home():
    return "<h1>Running Flask on Google Colab!</h1>"
{% endhighlight ruby %}
{% highlight ruby %}
@app.route('/regression', methods=['POST'])
def regression():
  age = float(request.json['Age'])
  potential = float(request.json['Potential'])
  inter = float(request.json['International Reputation'])
  weak_ft = float(request.json['Weak Foot'])
  skill_mov = float(request.json['Skill Moves'])
  crossing = float(request.json['Crossing'])
  finishing = float(request.json['Finishing'])
  head_acc = float(request.json['HeadingAccuracy'])
  shoot_pass = float(request.json['ShortPassing'])
  volleys = float(request.json['Volleys'])
  dribling = float(request.json['Dribbling'])
  curve = float(request.json['Curve'])
  fk_acc = float(request.json['FKAccuracy'])
  long_pass = float(request.json['LongPassing'])
  ball_ctrl = float(request.json['BallControl'])
  acc = float(request.json['Acceleration'])
  sprint_s = float(request.json['SprintSpeed'])
  agl = float(request.json['Agility'])
  react = float(request.json['Reactions'])
  balance = float(request.json['Balance'])
  shott_pow = float(request.json['ShotPower'])
  jumping = float(request.json['Jumping'])
  stamina = float(request.json['Stamina'])
  strength = float(request.json['Strength'])
  longshot = float(request.json['LongShots'])
  agg = float(request.json['Aggression'])
  interc = float(request.json['Interceptions'])
  poss = float(request.json['Positioning'])
  vision = float(request.json['Vision'])
  penalties = float(request.json['Penalties'])
  composure = float(request.json['Composure'])
  mark = float(request.json['Marking'])
  stand_tckl = float(request.json['StandingTackle'])
  slid_tckl = float(request.json['SlidingTackle'])
  gk_div = float(request.json['GKDiving'])
  gk_handl = float(request.json['GKHandling'])
  gk_kick = float(request.json['GKKicking'])
  gk_pos = float(request.json['GKPositioning'])
  gk_ref = float(request.json['GKReflexes'])
  real_f = float(request.json['Real_Face'])
  right_f = float(request.json['Right_Foot'])
  major = float(request.json['Major_Nation'])

  loaded_model = load_model('/content/weights_best.h5')
  scaler_feature = pickle.load(open(save_scaled_feature, 'rb'))
  scaler_label = pickle.load(open(save_scaled_label, 'rb'))

  data_web = [[ 
  age, 
  potential,
  inter,
  weak_ft,
  skill_mov, 
  crossing, 
  finishing, 
  head_acc, 
  shoot_pass, 
  volleys, 
  dribling, 
  curve, 
  fk_acc, 
  long_pass, 
  ball_ctrl, 
  acc, 
  sprint_s, 
  agl, 
  react, 
  balance, 
  shott_pow, 
  jumping, 
  stamina, 
  strength, 
  longshot, 
  agg, 
  interc, 
  poss, 
  vision, 
  penalties, 
  composure, 
  mark, 
  stand_tckl, 
  slid_tckl,
  gk_div,
  gk_handl, 
  gk_kick, 
  gk_pos, 
  gk_ref,
  real_f, 
  right_f,
  major]]
  
  scaling_data_web = scaler_feature.transform(data_web)
  predict_model_web = loaded_model.predict(scaling_data_web)
  invers_predict_web = scaler_label.inverse_transform(predict_model_web)

  return jsonify({
      "Overall": str(invers_predict_web[0][0])
  })
{% endhighlight ruby %}

[![3cjovt.th.png](https://iili.io/3cjovt.th.png)](https://freeimage.host/i/3cjovt)

Berikut adalah hasil dari proses deploy ke web service.

TERIMAKASIH
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:




