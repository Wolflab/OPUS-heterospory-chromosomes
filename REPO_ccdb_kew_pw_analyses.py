#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# # CCDB, Kew, and pw10 databases

# In[2]:


# directories to where files are
pw_path = '/path/to/Herbarium/Fern_Chromosome_Database/'
out_dir = '/path/to/heterospory_chromosome_paper/'
q_dir = '/path/to/Scripts/Qgarden_chromosomes/'

# Read in Paul's 10 samples
pw = pd.read_csv(pw_path + 'Paul_chromosome_1C_additions.csv')
print(pw.shape)
print(pw.columns)
print('')
# cleaned chromosome count database: ccdb
ccdb = pd.read_csv(out_dir + 'ccdb_clean_mini.csv')
print(ccdb.shape)
print(ccdb.columns)
print('')
# kew database
kew = pd.read_csv(q_dir + 'Kew_1C_pg.csv')
print(kew.shape)
print(kew.columns)


# In[3]:


# ADD pw samples to kew and ccdb databases
# first, divide pw into two files: one that gets appended to the ccdb (pw_count), and
# one that gets appended to kew (pw_size)
pw_for_kew = pw[~pw['DNA Amount1C (pg)'].isna()]
pw_for_kew2 = pw_for_kew[['genus', 'species', 'DNA Amount1C (pg)']]
pw_for_kew2.rename({'species':'Species', 'genus':'Genus'}, axis=1, inplace=True)
print(pw_for_kew2.shape)
print(pw_for_kew2.columns)
print('')
pw_for_ccdb = pw[~pw['min_parsed_n'].isna()]
pw_for_ccdb2 = pw_for_ccdb[['genus', 'species', 'category', 'min_parsed_n']]
print(pw_for_ccdb2.shape)
print(pw_for_ccdb2.columns)


# # merge Kew and pw

# In[4]:


# merge kew and pw
kew_mini = kew[['Genus', 'Species', 'DNA Amount1C (pg)']]
print(kew_mini.shape)
kew_pw = kew_mini.append(pw_for_kew2)
print(kew_pw.shape)
print(kew_pw.columns)


# In[5]:


print(kew_pw.isna().sum())
print(kew_pw[kew_pw['Species'].isna()])


# In[6]:


kew_pw2 = kew_pw.rename({'Species':'species', 'Genus':'genus'}, axis=1)
print(kew_pw2.shape)
print(kew_pw2.columns)
kew_pw2.to_csv(out_dir + 'Kew_pw_1C_pg_mini.csv', index=False)


# # Merge ccdb and pw

# In[7]:


# merge ccdb and pw
ccdb_pw = ccdb.append(pw_for_ccdb2)
print(ccdb_pw.shape)
print(ccdb_pw.columns)


# In[8]:


print(ccdb_pw['category'].value_counts())


# In[9]:


# ccdb_pw: remove horsetails and where min_parsed_n is nan
ccdb_pw2 = ccdb_pw[(ccdb_pw['category']!='Horsetails\nHomosporous') & ~(ccdb_pw['min_parsed_n'].isna())]
print(ccdb_pw2.shape)
print(ccdb_pw2.columns)
# should be 132 horsetails and 10,035 missing min_parsed_n


# In[10]:


ccdb_pw2.to_csv(out_dir + 'ccdb_pw_noHorsetails_noMissingN.csv', index=False)


# # Sidetracked. 
# ### Looking at category column

# In[11]:


# merge to get category for kew_pw also minimum per genus etc etc
# get unique genus species category from the ccdb
ccdb_cat = ccdb_pw[['genus','species', 'category']]
print(ccdb_cat.shape)
ccdb_cat2 = ccdb_cat.drop_duplicates(keep='first')
print(ccdb_cat2.shape)


# In[12]:


# merge ccdb_cat2 to the kew_pw on the kew_pw so we get categories
kew_pw2 = kew_pw.rename({'Species':'species', 'Genus':'genus'}, axis=1)
kew_pw_cat = pd.merge(ccdb_cat2, kew_pw2, on=[ 'species'], how='right')
print(kew_pw_cat.isna().sum())
print(kew_pw_cat.shape)


# In[13]:


# merge ccdb_cat2 to the kew_pw on the kew_pw so we get categories
kew_pw2 = kew_pw.rename({'Species':'species', 'Genus':'genus'}, axis=1)
kew_pw_cat = pd.merge(ccdb_cat2, kew_pw2, on=['genus', 'species'], how='right')
print(kew_pw_cat.isna().sum())
print(kew_pw_cat.shape)


# In[14]:


missing = kew_pw_cat[kew_pw_cat['category'].isna()]
print(missing.head())


# In[15]:


print(missing['genus'].nunique())


# In[16]:


kew_mini_fam = kew[['Family','Genus', 'Species', 'DNA Amount1C (pg)']]
print(kew_mini_fam.shape)
kew_pw_fam = kew_mini_fam.append(pw_for_kew2)
print(kew_pw_fam.shape)
print(kew_pw_fam.columns)
kew_pw22 = kew_pw_fam.rename({'Species':'species', 'Genus':'genus'}, axis=1)
kew_pw_cat22 = pd.merge(ccdb_cat2, kew_pw22, on=['genus', 'species'], how='right')
print(kew_pw_cat22.isna().sum())
missing22 = kew_pw_cat22[kew_pw_cat22['category'].isna()]
print(missing22.head())
print(missing22['Family'].nunique())


# In[17]:


Family = missing22['Family'].unique().tolist()
print(Family[:20])


# ### done playing with category column:  
# ### Won't worry about the missing categories, since we only want those rows in kew where we have info in ccdb

# # Data for Graphs

# ## ccdb minimum n for each unique genus

# In[18]:


print(ccdb_pw2.dtypes)


# In[19]:


# For upper two graphs, want just min chromosome count for each unique genus
# Get one min for each genus: (8347, 8)
print(ccdb_pw2['genus'].nunique())
ccdb_pw2_mini = ccdb_pw2[['genus', 'species','category', 'min_parsed_n']]
genus_mins = ccdb_pw2.loc[ccdb_pw2.groupby(['genus'])['min_parsed_n'].idxmin()].reset_index(drop=True)
print(genus_mins.shape)
print(genus_mins.columns)


# In[20]:


# who had multiple genus??
hunh = genus_mins.groupby(['genus'])['species'].count().reset_index()
print(hunh.sort_values('species').tail())


# In[21]:


# These are from the pw database. Don't understand why they are in here!
print(genus_mins[genus_mins['genus']=='Selaginella'])
print('')
print(genus_mins[genus_mins['genus']=='Regnellidium'])
print('')
print(genus_mins[genus_mins['genus']=='Salvinia'])


# In[22]:


# try this again another way


# In[23]:


genus_mins22 = ccdb_pw2.groupby(['genus'])['min_parsed_n'].min().reset_index()
print(genus_mins22.shape)
print(genus_mins22.columns)

hunh2 = pd.merge(genus_mins22, genus_mins, on=['genus', 'min_parsed_n'], how='left')
print(hunh2.shape)
print(hunh2.columns)


# In[24]:


what = hunh2.groupby(['genus'])['species'].count().reset_index()
print(what.sort_values('species').tail())


# In[25]:


hunh2[hunh2['genus']=='Regnellidium']


# In[26]:


# don't know why the pw addition is not behaving correctly
final_ccdb_mins = hunh2.drop(6639, axis=0)
print(final_ccdb_mins.shape)


# In[27]:


final_ccdb_mins[final_ccdb_mins['genus']=='Regnellidium']


# In[28]:


print(final_ccdb_mins.columns)


# In[29]:


final_ccdb_mins.to_csv(out_dir + "ccdb_noHorsetails_1min_per_genus.csv", index=False)


# ## ccdb ALL minimum n for each unique genus

# In[30]:


print(ccdb_pw2.columns)


# In[31]:


# Get ALL possible minimum values for each genus
ccdb_pw2_mini = ccdb_pw2[['genus', 'species','category', 'min_parsed_n']]
all_genus_mins = ccdb_pw2_mini.merge(ccdb_pw2_mini.groupby(['genus'])['min_parsed_n'].min().reset_index())
print(all_genus_mins.shape)


# In[32]:


print(all_genus_mins['genus'].nunique())
print(all_genus_mins.columns)


# In[33]:


all_genus_mins.to_csv(out_dir + 'ccdb_noHorsetails_ALLmin_per_genus.csv',index=False)


# # All diploids within a genus using a 1.2 divisable ratio

# In[34]:


# Getting ALL chromosome counts for each genus as a list
diploids = ccdb_pw2.copy()
diploids.sort_values(['genus', 'min_parsed_n'], inplace=True)
diploids_list = diploids.groupby(['genus'])['min_parsed_n'].unique().reset_index()
print(diploids_list.head())


# In[35]:


# want to get all minimums so long as they are >= 1.2 when divied
def get_num_mins(x):
    my_list = []
    for i in x:
        if i < 1.2*(min(x)):
            my_list.append(i)
    return my_list

diploids_list['diploid_mins'] = diploids_list['min_parsed_n'].apply(get_num_mins)
print(diploids_list.shape)
print(diploids_list.columns)
print(diploids_list.head(10))


# In[36]:


# CHECKING OUT THE DIPLOID LIST JUST TO SEE WHAT WE HAVE:
# let's see how many items in the lists
diploids_list['len_list'] = diploids_list['diploid_mins'].apply(lambda x: len(x))
print(diploids_list['len_list'].sum())
mm = diploids_list[diploids_list['len_list']> 1]
print(mm.shape)
print(mm.head())
print(mm['len_list'].value_counts())


# In[37]:


# now expand this list so each genus has each unique diploid count as a separate row
dip_mini = diploids_list[['genus', 'diploid_mins']]
print(dip_mini.shape)
# explode the list to rows
tnt_dip = dip_mini.explode('diploid_mins').reset_index(drop=True)
print(tnt_dip.shape)
print(tnt_dip.head())


# In[ ]:





# In[38]:


# Now, merge back so we can reclaim the category columns etc. that we'll need for graphing
tnt = pd.merge(ccdb_pw2, tnt_dip, left_on=['genus', 'min_parsed_n'] , 
               right_on=['genus', "diploid_mins"],how='inner')
print(tnt.shape)
#print(tnt['HOW'].value_counts())
#left_only     269032
#both           98373
#right_only         0
#Name: HOW, dtype: int64
tnt.to_csv(out_dir + 'ccdb_cleaned_EVERY_min_1pt2_ratio.csv', index=False)


# In[39]:


print(tnt['genus'].nunique())
print(tnt['species'].nunique())
print('')
print(ccdb_pw2['genus'].nunique())
print(ccdb_pw2['species'].nunique())


# In[40]:


ccdb_pw2.isna().sum()


# In[41]:


print(tnt.columns)
print(tnt['genus'].nunique())
print(tnt.isna().sum())
print('here are the missing species columns')
print(tnt[tnt['species'].isna()])
print("here is the dataframe")
print(tnt.head(10))


# # merge all diploids  (tnt: 'ccdb_cleaned_EVERY_min_1pt2_ratio.csv) with kew

# In[42]:


print(kew_pw2.columns)
print(tnt.columns)

kew_tnt = pd.merge(tnt,kew_pw2, on=['genus','species'], how='inner')
print(kew_tnt.columns)
print(kew_tnt.shape)


# In[43]:


print(kew_pw2.shape)


# In[44]:


kew_tnt_mini = kew_tnt[['genus', 'species', 'category', 'min_parsed_n','DNA Amount1C (pg)' ]]
print(kew_tnt_mini.shape)
simple_kew_tnt = kew_tnt_mini.drop_duplicates(keep='first')
print(simple_kew_tnt.shape)


# In[45]:


simple_kew_tnt.to_csv(out_dir + 'lower_graphs_db.csv', index=False)


# # GRAPHS. 
# ## TOP graphs

# In[46]:


# final_ccdb_mins == "ccdb_noHorsetails_1min_per_genus.csv"
print(final_ccdb_mins.columns)
print(final_ccdb_mins.isna().sum())
print('')

final_ccdb_mins.loc[(final_ccdb_mins["category"].str.contains("Heterosporous")), 'he_hom'] = "Heterosporous" 
final_ccdb_mins.loc[(final_ccdb_mins["category"].str.contains("Homosporous")), 'he_hom'] = "Homosporous"
hom = final_ccdb_mins[final_ccdb_mins['he_hom']=='Homosporous']
het = final_ccdb_mins[final_ccdb_mins['he_hom']=='Heterosporous']
print(genus_mins.shape)
print(hom.shape)
print(het.shape)


# ## bottom graphs

# In[47]:


print(simple_kew_tnt.shape)
print(simple_kew_tnt.columns)
print(simple_kew_tnt.isna().sum())
print('')

simple_kew_tnt.loc[(simple_kew_tnt["category"].str.contains("Heterosporous")), 'he_hom'] = "Heterosporous" 
simple_kew_tnt.loc[(simple_kew_tnt["category"].str.contains("Homosporous")), 'he_hom'] = "Homosporous"
bof_hom =simple_kew_tnt[simple_kew_tnt['he_hom']=='Homosporous']
bof_het = simple_kew_tnt[simple_kew_tnt['he_hom']=='Heterosporous']

print(bof_hom.shape)
print(bof_het.shape)


# In[48]:



# And now graph again:

my_order = ['Angiosperms\nHeterosporous', "Gymnosperms\nHeterosporous",
           "Ferns\nHeterosporous", "Ferns\nHomosporous",
           "Lycophytes\nHeterosporous", "Lycophytes\nHomosporous"]


sns.set_style("white")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(15,18))
# upper left

#g = sns.boxplot(x='category', y='n', data=ccdb_pw, palette=["lightsteelblue"], order=my_order,linewidth=0.5, showfliers = False, ax=ax1)
my_colors = ['#fc8d62', '#8da0cb']
g = sns.boxplot(x='category', y='min_parsed_n', data=final_ccdb_mins, palette=my_colors, hue='he_hom',order=my_order,linewidth=0.5, showfliers = False, ax=ax1)
legend = ax1.legend(fontsize = 13.5,bbox_to_anchor=(1.004,0.96))
#legend.texts[0].set_text("Whatever else")
ax1.set_ylabel('min n per genus', fontsize=14)
ax1.set_xlabel('')
ax1.set_xticklabels(labels=my_order ,rotation=50, fontsize=12)
ax1.grid(True, axis='y')

# upper right
p1=sns.kdeplot(data=hom, x="min_parsed_n", color='#8da0cb', ax=ax2, label='Homosporous')
p1.set(xlim=(-15, 150))
ax22 = ax2.twinx()
ax2.grid(False)
p2 = sns.kdeplot(data=het, x="min_parsed_n", ax=ax22, color='#fc8d62', label='Heterosporous')

ax2.set_xlabel('n', fontsize=12)
ax2.set_ylabel('Homosporous density', fontsize=12)
ax22.set_ylabel('Heterosporous density', fontsize=12)
ax2.legend(loc = (.55,.85), frameon = False, fontsize=14)
ax22.legend( loc = (.55, .8), frameon = False, fontsize=14)

# bottom left
g2 = sns.boxplot(x='category', y='DNA Amount1C (pg)', data=simple_kew_tnt, palette=my_colors, hue= 'he_hom',order=my_order,linewidth=0.5, showfliers = False, ax=ax3)
legend = ax3.legend(fontsize = 13.5,bbox_to_anchor=(1.004,0.96))
#loc='upper right'
ax3.set_ylabel('1C (pg) for same spp. for n', fontsize=14)
ax3.set_xlabel('')
ax3.set_xticklabels(labels=my_order ,rotation=50, fontsize=12)
ax3.grid(True, axis='y')

#bottom right
pp=sns.kdeplot(data=bof_hom, x="DNA Amount1C (pg)", color='#8da0cb', ax=ax4, label='Homosporous')
#pp.set(xlim=(-15, 150))
ax44 = ax4.twinx()
ax4.grid(False)
pp2 = sns.kdeplot(data=bof_het, x="DNA Amount1C (pg)", ax=ax44, color='#fc8d62', label='Heterosporous')

ax4.set_xlabel('DNA Amount1C (pg)', fontsize=12)
ax4.set_ylabel('Homosporous density', fontsize=12)
ax44.set_ylabel('Heterosporous density', fontsize=12)
ax4.legend(loc = (.55,.85), frameon = False, fontsize=14)
ax44.legend( loc = (.55, .8), frameon = False, fontsize=14)

#ax1.text(-0.3, 53, "A",  size=30, weight='bold', zorder=4)
#ax2.text(-9.0, 0.0288, "B", size=30, weight='bold', zorder=5)
#ax3.text(-0.3, 31, "C", size=30, weight='bold', zorder=6)
#ax4.text(-26, 0.031, "D", size=30, weight='bold', zorder=7)
#fig.savefig(out_dir + 'hetero_chromo_num_paper_withPWdata_ALLmins_ratio_1pt2_noABCD.png', bbox_inches='tight', dpi=800)


# # STATS

# In[49]:


# top: genus_mins
print('UPPER GRAPHS: chromosome counts')
print('')
print('CATEGORY')
print(final_ccdb_mins['category'].value_counts())
print('')
print('HOMOSPOROUS')
print(hom['min_parsed_n'].describe())
print('')
print('HETEROSPOROUS')
print(het['min_parsed_n'].describe())
print('')
print('BOTH')
print(final_ccdb_mins['min_parsed_n'].describe())


# In[50]:


#bottom: kew_tnt
print('LOWER GRAPHS: genome size')
print('')
print('CATEGORY')
print(simple_kew_tnt['category'].value_counts())
print('')
print('HOMOSPOROUS')
print(bof_hom['DNA Amount1C (pg)'].describe())
print('')
print('HETEROSPOROUS')
print(bof_het['DNA Amount1C (pg)'].describe())
print('')
print('BOTH')
print(simple_kew_tnt['DNA Amount1C (pg)'].describe())


# # ended here!!!

# In[214]:


# all ferns
# homosporous ferns - Ophioglossaceae
# using the lower_graphs_db.csv
print(simple_kew_tnt.columns)
print(simple_kew_tnt['category'].value_counts())


# In[213]:


only_ferns = simple_kew_tnt[simple_kew_tnt['category'].str.contains('Ferns')]
print(only_ferns.shape)


# In[218]:


sns.relplot(x="min_parsed_n", y="DNA Amount1C (pg)", hue="category",data=only_ferns)


# In[258]:


ophio = ccdb_pw2[ccdb_pw2['family']=='Ophioglossaceae']
print(ophio.shape)
oph_genera = ophio['genus'].unique().tolist()
oph_gen =oph_genera + ['Psilotum', 'Tmesipteris']
print(len(oph_gen))
print(oph_gen)


# In[259]:


only_ferns['category2'] = only_ferns['category']
print(only_ferns.head())


# In[264]:


#df['my_channel'] = np.where(df.my_channel > 20000, 0, df.my_channel)

only_ferns['category2'] = np.where(only_ferns['genus'].isin(oph_gen), 'Oph+Psil', only_ferns['category'])
print(only_ferns.shape)
print(only_ferns['category2'].value_counts())


#fern_no_oph = only_ferns[~only_ferns['genus'].isin(oph_gen)]
#print(fern_no_oph.shape)
#oph_fern = only_ferns[only_ferns['genus'].isin(oph_genera)]
#print(oph_fern)


# In[270]:


only_ferns['min_parsed_n'] = only_ferns['min_parsed_n'].astype(float)
only_ferns.dtypes


# In[271]:


sns.lmplot(x="min_parsed_n", y="DNA Amount1C (pg)", hue="category2",data=only_ferns)


# In[273]:


fern_hom = only_ferns[only_ferns['category2']=='Ferns\nHomosporous']
print(fern_hom.shape)


# In[289]:


my_colors = ['#8da0cb', '#fc8d62','purple']
g = sns.lmplot(x="min_parsed_n", y="DNA Amount1C (pg)", hue="category2",data=only_ferns, fit_reg=False, legend=False, palette=my_colors)
sns.regplot(x="min_parsed_n", y="DNA Amount1C (pg)",data=only_ferns, scatter=False, ax=g.axes[0, 0], color='dimgrey')
sns.regplot(x="min_parsed_n", y="DNA Amount1C (pg)",data=fern_hom, scatter=False, ax=g.axes[0, 0], color='#8da0cb')
plt.xlabel("genus min n", size=14)
plt.ylabel("DNA Amount1C (pg)", size=14)
plt.legend(title='', fontsize=12)

plt.savefig(out_dir + 'min_parsed_n_vs_1C.png', dpi=800, bbox_inches='tight')


# In[255]:


oopsy = fern_no_oph[fern_no_oph['DNA Amount1C (pg)']>50].reset_index()

oopsy_list = oopsy['genus'].unique().tolist()
print(oopsy_list)
fern_no_oph2 = fern_no_oph[~fern_no_oph['genus'].isin(oopsy_list)]
print(fern_no_oph2.shape)


# In[294]:


# get all merges
cr_ccdb_pw2 = ccdb_pw2[['genus','species', 'min_parsed_n','category']]
cr_kew_pw2 =kew_pw2[['genus','species','DNA Amount1C (pg)']]
print(cr_ccdb_pw2.shape)
print(cr_kew_pw2.shape)
all_merges = pd.merge(cr_ccdb_pw2, cr_kew_pw2, on=['genus','species'], how='inner')
print(all_merges.shape)

'''
left_only     242821
both          145477
right_only      4053
'''


# In[295]:


cr_every_fern = all_merges[all_merges['category'].str.contains('Fern')]
print(cr_every_fern.shape)
cr_every_fern['category2'] = np.where(cr_every_fern['genus'].isin(oph_gen), 'Oph+Psil', cr_every_fern['category'])
print(cr_every_fern.shape)
print(cr_every_fern['category2'].value_counts())


# In[297]:


cr_fern_hom = cr_every_fern[cr_every_fern['category2']=='Ferns\nHomosporous']
print(cr_fern_hom.shape)


# In[299]:


my_colors = ['#8da0cb', '#fc8d62','purple']
g = sns.lmplot(x="min_parsed_n", y="DNA Amount1C (pg)", hue="category2",data=cr_every_fern, fit_reg=False, legend=False, palette=my_colors)
sns.regplot(x="min_parsed_n", y="DNA Amount1C (pg)",data=cr_every_fern, scatter=False, ax=g.axes[0, 0], color='dimgrey')
sns.regplot(x="min_parsed_n", y="DNA Amount1C (pg)",data=cr_fern_hom, scatter=False, ax=g.axes[0, 0], color='#8da0cb')
plt.xlabel("n", size=14)
plt.ylabel("DNA Amount1C (pg)", size=14)
plt.legend(title='', fontsize=12)

plt.savefig(out_dir + 'n_vs_1C.png', dpi=800, bbox_inches='tight')


# In[301]:


from scipy import stats


# In[317]:


# mins regression stats: min_parsed_n_vs_1C.png
print('minimums from ferns: min_parsed_n_vs_1C.png')
slope, intercept, r_value, p_value, std_err = stats.linregress(only_ferns['min_parsed_n'],only_ferns['DNA Amount1C (pg)'])
print('slope: {:.3f}, intercept: {:.3f}, r_value: {:.3f}, p_value: {:.3f}, std_error: {:.3f}'.format(slope, intercept, r_value, p_value, std_err))


# In[318]:


print('minimums from ferns homosporous: min_parsed_n_vs_1C.png')
slope, intercept, r_value, p_value, std_err = stats.linregress(fern_hom['min_parsed_n'],fern_hom['DNA Amount1C (pg)'])
print('slope: {:.3f}, intercept: {:.3f}, r_value: {:.3f}, p_value: {:.3f}, std_error: {:.3f}'.format(slope, intercept, r_value, p_value, std_err))


# In[319]:


# n (all) regression stats
print('all: ferns: n_vs_1C.png')
slope, intercept, r_value, p_value, std_err = stats.linregress(cr_every_fern['min_parsed_n'],cr_every_fern['DNA Amount1C (pg)'])
print('slope: {:.3f}, intercept: {:.3f}, r_value: {:.3f}, p_value: {:.3f}, std_error: {:.3f}'.format(slope, intercept, r_value, p_value, std_err))


# In[320]:


print('all: ferns-homosporous: n_vs_1C.png')
slope, intercept, r_value, p_value, std_err = stats.linregress(cr_fern_hom['min_parsed_n'],cr_fern_hom['DNA Amount1C (pg)'])
print('slope: {:.3f}, intercept: {:.3f}, r_value: {:.3f}, p_value: {:.3f}, std_error: {:.3f}'.format(slope, intercept, r_value, p_value, std_err))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[293]:


cc_noBryo_pw.to_csv(out_dir + 'ccdb_pw10_noBryo.csv' , index=False)


# In[268]:


out_dir = '/Users/carol/Dropbox/Paul_and_Carol_Shared/heterospory_chromosome_paper/'
genus_mins.to_csv(out_dir + 'ccdb_noBryo_1min_per_genus.csv' ,index=False)


# In[274]:


all_genus_mins.to_csv(out_dir + 'ccdb_noBryo_ALLmin_per_genus.csv',index=False)


# In[284]:


kew_pw.to_csv(out_dir + 'Kew_pw10_1C_pg.csv', index=False)


# In[ ]:


tnt.to_csv(out_dir + 'ccdb_cleaned_EVERY_min_1pt2_ratio.csv', index=False)


# In[427]:


dummy = pd.DataFrame({'a': [[10.0, 11.0, 12.0, 14.0, 16.0, 19.0, 20.0], [10.0, 11.0, 14.0, 14.0, 16.0, 19.0, 20.0],[10.0,11.0], [2.0,3.0]]})
print(dummy)


# In[432]:


def get_num_mins(x):
    my_list = []
    for i in x:
        if i < 1.2*(min(x)):
            my_list.append(i)
    return my_list
dummy['num_mins'] = dummy['a'].apply(get_num_mins)
print(dummy)


# In[ ]:




