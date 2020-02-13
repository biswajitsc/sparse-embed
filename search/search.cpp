#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

using namespace std;

typedef vector<float> Vec;
typedef vector<pair<int,float> > SparseVec;
typedef int Label;

typedef pair<Label, SparseVec> Instance;
typedef vector<SparseVec> InvertedIndex;

const int QUERY_SIZE = 10000;
const int NUM_INS_PER_PRINT = 100000;

int TOP_R = 1;
float CONF_THD = 0.25f;


void read_embed_data(char* embed_fpath, vector<Instance*>& embed_data, int& dim, int& num_class, int limit_N = 10000000){
   
    ifstream fin(embed_fpath, ios::in | ios::binary);
    int N;
    fin.read( (char*) &N, sizeof(int) );
    N = min( N, limit_N );

    cerr << "Reading " << N << " instances..." << endl;

    int label_offset = num_class;
    embed_data.resize(N);
    for(int i=0;i<N;i++){
        embed_data[i] = new Instance();
        int lab, nnz;
        fin.read((char*) &lab, sizeof(int));
        lab = label_offset + lab;

        embed_data[i]->first = lab;
        if(lab+1 > num_class) num_class = lab+1;
        fin.read((char*) &nnz, sizeof(int));
        SparseVec& embed = embed_data[i]->second;
        embed.resize(nnz);
        for(int j=0;j<nnz;j++){
            fin.read((char*) &(embed[j].first), sizeof(int));
            if(embed[j].first+1 > dim)
            	dim = embed[j].first+1;
        }
        for(int j=0;j<nnz;j++){
            fin.read((char*) &(embed[j].second), sizeof(float));
        }	
        if(i % NUM_INS_PER_PRINT == 0 && i!=0)
            cerr << i << " instances read." << endl;
    }
}


float inner_prod(SparseVec& a, SparseVec& b){
    SparseVec::iterator it1 = a.begin();
    SparseVec::iterator it2 = b.begin();
    float sum = 0.0f;
    while(it1!=a.end() && it2!=b.end()){
        if( it1->first < it2->first )
                it1++;
        else if( it1->first > it2->first )
                it2++;
        else{
                sum += it1->second * it2->second;
                it1++;
                it2++;
        }
    }
    return sum;
}


void split_into_query_db(vector<Instance*>& data, int query_size, 
        vector<Instance*>& query_list, vector<Instance*>& db_list){		
    //Assume the data is already shuffled.
    // random_shuffle(data.begin(), data.end());
    query_list.clear();
    for(int i=0;i<query_size;i++)
        query_list.push_back(data[i]);
    for(int i=0;i<data.size();i++)
        db_list.push_back(data[i]);
}


bool sortbysec(const pair<int,float> &a, const pair<int,float> &b){ 
    return (a.second > b.second); 
} 


int topk = 1000;

class SearchEngine {
    public:
    SearchEngine(vector<Instance*>& data, int _dim){
        N = data.size();
        db = data;
        dim = _dim;
        build_index(db, dim);
        score_vec.resize(N);
        rank_list.resize(N);
        dense_query.resize(dim);
    }

    void build_index(vector<Instance*>& data, int dim){
        int N = data.size();
        long nnz = 0, nnz_thd = 0;
        db_index.resize(dim);
        for(int i=0;i<N;i++){
            SparseVec& embed = data[i]->second;
            for(auto& p: embed){
                db_index[p.first].push_back(make_pair(i,p.second));
                nnz++;
            }
        }
        cerr << "Index built. NNZ% = " << ((float)nnz/N/dim*100.0) << endl;

        mata.resize(N, vector<float>(dim));
        for (size_t i=0; i<N;i++)
            for (size_t j=0; j<dim;j++)
                mata[i][j] = (float)(rand() % 101) / 101.1;
        
        // Filling the dense dense query vector with random floats.
        // This is done to simulate the dense reranking step without loading
        // the dense vectors.
        for (size_t i=0; i<dense_query.size(); i++)
            dense_query[i] = (float)(rand() % 101) / 101.1;
    }
    
    int search(SparseVec query) {
        memset((void*) &(score_vec[0]), 0, score_vec.size()*sizeof(float));
            for(auto& p: query){
                for(auto& p2: db_index[p.first]){
                    score_vec[p2.first] += p.second*p2.second;
                }
            }
        // ************************************ Comment below for no reranking ****************
        int rank_list_size = 0;
        for(int i=0;i<score_vec.size();i++){
            if(score_vec[i] > CONF_THD)
                rank_list[rank_list_size++] = make_pair(i,score_vec[i]);
        }
        if(rank_list_size >= topk)
            nth_element(rank_list.begin(), rank_list.begin()+topk-1, rank_list.begin() + rank_list_size, sortbysec);
        int max_iter = min(rank_list_size, topk);

        for (size_t j=0; j<max_iter; j++)
        {
            int idx = rank_list[j].first;
            rank_list[j].second = 0;
            for (size_t i = 0; i<dim;i++)
                rank_list[j].second += mata[idx][i] * dense_query[i];
        }
        if(max_iter > TOP_R)
            nth_element(rank_list.begin(), rank_list.begin()+TOP_R-1, rank_list.begin()+max_iter-1, sortbysec);
        return max_iter;
        // *********************************** Comment ends here ****************************
        return 0;
    }
    

    private:
    int N;
    int dim;
    vector<Instance*> db;
    InvertedIndex db_index;
    vector<float> score_vec;
    vector<vector<float> > mata;
    vector< pair<int, float> > rank_list;
    vector<float> dense_query;
};


int main(int argc, char** argv){
    if( argc-1 < 3 ){
        cerr << "Command line arguments error" << endl;
        cerr << "Usage: ./search_test <embed_fpath> <top_k_rerank> <thr> [queryset_fpath]" << endl;
        exit(0);
    }
    srand(1);
    char* embed_fpath = argv[1];
    topk = atoi(argv[2]);
    CONF_THD = atof(argv[3]);
    cerr << "Confidence thr " << CONF_THD << endl;
    TOP_R = 1;
    char* queryset_fpath = NULL;
    if( argc-1 > 3 ){
        queryset_fpath = argv[4];
    }

    //Read Embeddings
    vector<Instance*> embed_data;
    int dim=-1, num_class=0;
    read_embed_data(embed_fpath, embed_data, dim, num_class);
    cerr << "N=" << embed_data.size() << ", D=" << dim << ", #class=" << num_class << endl;
    int num_nonquery_class = num_class;
    
    vector<Instance*> queryset_data;
    if(queryset_fpath!=NULL){
        read_embed_data(queryset_fpath, queryset_data, dim, num_class);
        cerr << "Nq=" << queryset_data.size() << ", D=" << dim << ", #class=" << num_class << endl;
    }
    
    //Split into Query and DB
    vector<Instance*> query_list, db_list;
    if(queryset_fpath==NULL) {
        int i;
        for(i=0; i<QUERY_SIZE; i++) query_list.push_back(embed_data[i]);
        for(;i<embed_data.size(); i++) db_list.push_back(embed_data[i]);
        cerr << "Splitting into " << (int) query_list.size() << " queries and a database of size " << (int) db_list.size() << endl;
    }
    else {
    query_list.insert(query_list.end(), queryset_data.begin(), queryset_data.end());
    db_list.insert(db_list.end(), embed_data.begin(), embed_data.end());
    cerr << "Splitting into " << (int) query_list.size() << " queries and a database of size " << (int) db_list.size() << endl;
    }

    //Build Inverted Index
    cerr << "Building index..." << endl;
    SearchEngine* engine = new SearchEngine(db_list, dim);

    vector<int> num_ins(num_class, 0);
    for(int i=0;i<queryset_data.size();i++){
        Instance* ins = queryset_data[i];
        num_ins[ins->first]++;
    }

    //Search
    double start = omp_get_wtime();
    int Nt = query_list.size();
    
    int empty_count=0;
    int max_rank_len = -1;
    long cum_rank_len = 0;
    for(int i=0;i<query_list.size();i++){
        Instance* q = query_list[i];
        int q_lab = q->first;
        SparseVec& embed = q->second;
        int rank_list_size = engine->search(embed);
        cum_rank_len += rank_list_size;
        max_rank_len = max(max_rank_len, rank_list_size);
     }
     double end = omp_get_wtime();

    float avg_rank_len = cum_rank_len / Nt;
    cerr << "Max reranking list length = " << max_rank_len << endl;
    cerr << "Average reranking list length = " << avg_rank_len << endl;
    cerr << "Total search time = " << end-start << "s" << endl;
    cerr << "Search Time per Query = " << 1e3*(end-start)/Nt << "ms" << endl;

   return 0;	 
}
