#include <bits/stdc++.h>
#include <iostream>
using namespace std;

const int THE_DIVIDE=24; // 笔画和背景的临界。
const int MAX_WORD=200; // 最大字所占像素点。
const int BLOCK=100; //一个字中最大的空白（二）

#define re int
inline int read(){
	int x=0,ff=1;char c=getchar();
	while(c<'0'||c>'9'){if(c=='-')ff=-1;c=gethchar();}
	while(c>='0'&&c<='9'){x=(x<<1)+(x<<3)+(c^'0');c=getchar();}
	return x*ff;
}

bool findlr(vector<bool > b,int l,int r){
	for(re i=l;i<=r;i++){
		if(b[i])return 1;
	}
	return 0;
}
vector<pair<int,int> >  divide_h(int n,int m,vector<vector<int> > d){
	
	vector<vector<bool> > a;
	vector<bool> _;
	for(re i=0;i<n;i++){
		a[i].emplace_back(_);
		for(re j=0;j<m;j++){
			a[i].emplace_back((d[i][j]>THE_DIVIDE));
		}
	}

	vector<pair<int,int> > qwq;
	for(re i=0;i<n;i++){
		int ex=-1;
		for(re j=0;j<m;j++){
			if(a[i][j]) {ex=j;break;}
		}
		if(ex==-1) continue;
		int l=max(ex-MAX_WORD,0),r=min(ex+MAX_WORD,m);
		int u = i + 1, la = i;
		while(u<n){
			if(findlr(a[u],l,r)){
				la=u;u++;continue;
			}
			if(u-la>BLOCK||!findlr(a[u],0,m-1))break;
			u++;
		}
		qwq.emplace_back(make_pair(i,u-1));
		i=u-1;
	}
	return qwq;
}
/*
基础版本，大概率需要改进。
取a可以改为利用周围值加权。
*/
