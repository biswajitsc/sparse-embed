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

//#define WRITE_MODE

typedef vector<float> Vec;
typedef vector<pair<int,float> > SparseVec;
typedef int Label;

typedef pair<Label, SparseVec> Instance;

typedef vector<SparseVec> InvertedIndex;

const int QUERY_SIZE = 10000;
const int NUM_INS_PER_PRINT = 100000;

const float NNZ_THD = 0.005f;
int TOP_R = 1;
//const float CONF_THD = 3.0f;
// const float CONF_THD = 0.7f;
float CONF_THD = 0.25f;


void read_embed_data( char* embed_fpath, vector<Instance*>& embed_data, int& dim, int& num_class, int limit_N = 10000000 ){
   
   ifstream fin( embed_fpath, ios::in | ios::binary );
	 int N;
	 fin.read( (char*) &N, sizeof(int) );
   N = min( N, limit_N );

	 cerr << "Reading " << N << " instances..." << endl;
   
	 int label_offset = num_class;
	 embed_data.resize(N);
	 for(int i=0;i<N;i++){
			 
			 embed_data[i] = new Instance();

			 int lab, nnz;
			 fin.read( (char*) &lab, sizeof(int) );
			 lab = label_offset + lab;
       
			 embed_data[i]->first = lab;
       if( lab+1 > num_class )
					 num_class = lab+1;
       
			 fin.read( (char*) &nnz, sizeof(int) );
			 SparseVec& embed = embed_data[i]->second;
       embed.resize(nnz);
			 
			 for(int j=0;j<nnz;j++){
					fin.read(  (char*) &(embed[j].first), sizeof(int) );
					if( embed[j].first+1 > dim )
							dim = embed[j].first+1;
			 }
       for(int j=0;j<nnz;j++){
					fin.read(  (char*) &(embed[j].second), sizeof(float) );
			 }
			 
			 if( i % NUM_INS_PER_PRINT == 0 && i!=0 )
					 cerr << i << " instances read." << endl;
	 }
}


float inner_prod( SparseVec& a, SparseVec& b ){

		SparseVec::iterator it1 = a.begin();
		SparseVec::iterator it2 = b.begin();

		float sum = 0.0f;
		while( it1!=a.end() && it2!=b.end() ){
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


void write_to_file( char* out_fpath, vector<Instance*>& data, int dim ){
   
		ofstream fout(out_fpath);
		fout << data.size() << " " << dim << " " << CONF_THD << endl;
		for(int i=0;i<data.size();i++){
				SparseVec& embed = data[i]->second;
				fout << embed.size() << " ";
				for(auto& p: embed)
						fout << p.first << ":" << p.second << " ";
				fout << endl;

				if( i%100000==0 )
						cerr << "writing " << i << " instances" << endl;
		}
		fout.close();
}


void split_into_query_db(vector<Instance*>& data, int query_size, 
				vector<Instance*>& query_list, vector<Instance*>& db_list){
		
		//Assume the data is already shuffled.
    //random_shuffle(data.begin(), data.end());
    
		query_list.clear();
		for(int i=0;i<query_size;i++)
				query_list.push_back(data[i]);
		//for(int i=query_size;i<data.size();i++)
		//		db_list.push_back(data[i]);
		for(int i=0;i<data.size();i++)
				db_list.push_back(data[i]);
}


float inner_prod(Vec& v, SparseVec& sv){
    
		float sum = 0.0f;
		for(auto& p: sv){
			 sum += v[p.first]*p.second;
		}
		return sum;
}


bool sortbysec(const pair<int,float> &a, const pair<int,float> &b){ 
		    return (a.second > b.second); 
} 


double sort_time = 0.0;
double acc_time = 0.0;
double d2s_time = 0.0;
int topk = 100;
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
								db_index[p.first].push_back( make_pair(i,p.second) );
								nnz++;
						}
				}
				cerr << "Index built. nnz=" << nnz << ", sparsity ratio=" << ((float)nnz/N/dim) << endl;
                
                mata.resize(N, vector<float>(dim));
                for (size_t i=0; i<N;i++)
                    for (size_t j=0; j<dim;j++)
                        mata[i][j] = (float)(rand() % 101) / 101.1;
				
				for (size_t i=0; i<dense_query.size(); i++)
					dense_query[i] = (float)(rand() % 101) / 101.1;
		}
		
		int search(SparseVec query) {
        
        //make binary
				// for(auto& p: query){
				// 		p.second = 1.0;
				// }

        memset( (void*) &(score_vec[0]), 0, score_vec.size()*sizeof(float) );
        
				// acc_time -= omp_get_wtime();
				for(auto& p: query){
						for(auto& p2: db_index[p.first]){
                score_vec[p2.first] += p.second*p2.second;
						}
				}
				// acc_time += omp_get_wtime();
        
				// d2s_time -= omp_get_wtime();
			// ************************************ Comment below for no reranking ****************
		// 		int rank_list_size = 0;
        // for(int i=0;i<score_vec.size();i++){
		// 				if( score_vec[i] > CONF_THD )
		// 						rank_list[rank_list_size++] = make_pair(i,score_vec[i]);
        // }
		// // cerr << "rank list size " << rank_list_size << endl;
        // // d2s_time += omp_get_wtime();

        //         if( rank_list_size >= topk ){
		// 				// sort_time -= omp_get_wtime();
		// 				nth_element( rank_list.begin(), rank_list.begin()+topk-1, rank_list.begin() + rank_list_size, sortbysec );
		// 				// sort(rank_list.begin(), rank_list.begin()+TOP_R+1, sortbysec);
		// 				//sort(rank_list.begin(), rank_list.end(), sortbysec);
		// 		}
		// 		// cerr << "threshold value" << rank_list[topk - 1].second << endl;
        //         int max_iter = min(rank_list_size, topk);
				
        //         for (size_t j=0; j<max_iter; j++)
        //         {
		// 			int idx = rank_list[j].first;
		// 			rank_list[j].second = 0;
        //             for (size_t i = 0; i<dim;i++)
        //                 //ans2(j) += mat(j,i) * vec(i);
        //                 rank_list[j].second += mata[idx][i] * dense_query[i];
        //                 //ans2a[j] += mata[j][i] * veca[i];
        //         }
		// return max_iter;
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
      cerr << "./search_test [embed_fpath] [top_?] [thr] (queryset_fpath)" << endl;
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
	 if( queryset_fpath!=NULL ){
			 read_embed_data(queryset_fpath, queryset_data, dim, num_class);
			 cerr << "Nq=" << queryset_data.size() << ", D=" << dim << ", #class=" << num_class << endl;
	 }
	 
	// Truncate small non-zero values
	//  truncate( embed_data, NNZ_THD, dim );
	//  if( queryset_fpath!=NULL )
	//     truncate( queryset_data, NNZ_THD, dim );
	 
   	// Binarize
	//  binarize( embed_data );
	//  if( queryset_fpath != NULL )
	// 		 binarize( queryset_data );
   
	// Normalizing
	// cerr << "Embedding normalizing..." << endl;
	// normalize( embed_data );
   	// if( queryset_fpath!=NULL )
	// 		 normalize( queryset_data );
   
	 //Quantization (optional)
	 /*int bit_width = 7;
	 int init_pos = 0;
	 quantize( embed_data, bit_width, init_pos );
	 if( queryset_fpath!=NULL )
			 quantize( queryset_data, bit_width, init_pos );*/
   
	 //write
#ifdef WRITE_MODE
	 //char* fname = "mscele5.8M_embed_bin.txt";
	 char* fname = "imgnet7.8M_embed_bin.txt";
	 cerr << "writing to file..." << fname << endl;
	 write_to_file( fname, embed_data, dim );
#endif

	 //Split into Query and DB
	 vector<Instance*> query_list, db_list;
	 if( queryset_fpath==NULL ){
	    //  split_into_query_db( embed_data, QUERY_SIZE, query_list, db_list );
		 int i;
		 for(i=0; i<QUERY_SIZE; i++) query_list.push_back(embed_data[i]);
		 for(;i<embed_data.size(); i++) db_list.push_back(embed_data[i]);
		 cerr << "Splitting into " << (int) query_list.size() << " queries and a database of size " << (int) db_list.size() << endl;
	 }else{
	   query_list.insert(query_list.end(), queryset_data.begin(), queryset_data.end());
	   db_list.insert(db_list.end(), embed_data.begin(), embed_data.end());
	   cerr << "Splitting into " << (int) query_list.size() << " queries and a database of size " << (int) db_list.size() << endl;
	 }
   
   //Build Inverted Index
	 cerr << "Building index..." << endl;
   SearchEngine* engine = new SearchEngine( db_list, dim );
   //SearchEngine* engine = new JaccardSearchEngine( db_list, dim );
   
	 vector<int> num_ins(num_class, 0);
   for(int i=0;i<queryset_data.size();i++){
			Instance* ins = queryset_data[i];
      num_ins[ins->first]++;
	 }
   
	 //Search
#ifdef WRITE_MODE
	 ofstream fout_q("query.txt");
	 ofstream fout_r("result.txt");
   for(int i=0;i<db_list.size();i++)
			 db_list[i]->first = i;
#endif
	 
	 double start = omp_get_wtime();
	 int Nt = query_list.size();
	 
	 int empty_count=0;
	 int max_rank_len = -1;
	 float prec_avg=0.0f;
	 long cum_hit = 0;
	 long optimal_hit_count = 0;
	 long cum_rank_len = 0;
	 for(int i=0;i<query_list.size();i++){

			 Instance* q = query_list[i];

			 int q_lab = q->first;
			 SparseVec& embed = q->second;

#ifdef WRITE_MODE
			 if(i<10){
					 fout_q << embed.size() << " ";
					 for(auto& p: embed)
							 fout_q << p.first << ":" << p.second << " ";
					 fout_q << endl;
			 }
#endif
			 
			//  vector<pair<Instance*,float> > rank_list;
			int rank_list_size = engine->search( embed);
       if( rank_list_size < topk ){
					 //cerr << "return size " << rank_list.size() << " smaller than " << TOP_R << endl;
					 //exit(0);
					 empty_count++;
			 }
			 cum_rank_len += rank_list_size;
			 max_rank_len = max( max_rank_len, rank_list_size);

#ifdef WRITE_MODE
			 if(i<10){
					 fout_r << rank_list.size() << " ";
					 for(auto& p: rank_list){
							 fout_r << p.first->first << ":" << p.second << " ";
					 }
					 fout_r << endl;
			 }
#endif
			 
			 //Evaluate
			//  int hit_count = 0;
			//  for(int r=1;r<rank_list.size();r++){
			// 		 Instance* ins  = rank_list[r].first;
			// 		 //if( ins->first >= num_nonquery_class && ins->first != q_lab )
			// 			//	 continue;
					 
			// 		 int hit = (ins->first == q_lab);
					 
			// 		 if( r >= TOP_R+1 )
			// 				 break;

			// 		 //if( hit==0 )
			// 		//		 break;
					 
			// 		 hit_count += hit;
			//  }
			//  prec_avg += (float)hit_count/TOP_R;
			 //cum_hit += hit_count;
       //optimal_hit_count += (num_ins[q_lab]-1);
			 
			//  if( i % 100 == 0 )
			// 		 cerr << i << " processed..." << endl;
	 }
#ifdef WRITE_MODE
   fout_q.close();
	 fout_r.close();
#endif
	 
	 //prec_avg = (float)cum_hit / optimal_hit_count;
	//  prec_avg /= Nt;
	 double end = omp_get_wtime();

	float avg_rank_len = cum_rank_len / Nt;
	cerr << "Top k=" << topk << endl;
   cerr << "Search Time=" << end-start << " s." << endl;
	 cerr << "Search Time per Query=" << 1e3*(end-start)/Nt << " ms." << endl;
	 
	//  cerr << "Top-" << TOP_R << " Prec= " << prec_avg << endl;

	 cerr << "Average Ranklist length=" << avg_rank_len << endl;
	 cerr << "Max Ranklist length=" << max_rank_len << endl;
	 cerr << "#empty-cases=" << empty_count << endl;
   cerr << "sort_time per Query=" << 1e3*sort_time/Nt << " ms." << endl;
   cerr << "acc_time per Query=" << 1e3*acc_time/Nt << " ms." << endl;
   cerr << "d2s_time per Query=" << 1e3*d2s_time/Nt << " ms." << endl;

   return 0;	 
}
