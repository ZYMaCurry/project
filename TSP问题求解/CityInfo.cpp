#include "CityInfo.h"
//���ó�������
void CityInfo::SetName(string na){
    Name=na;
}
//���ó�������
void CityInfo::SetCityIndex(int index){
    CityIndex=index;
}
//����Coordx
void CityInfo::SetCoordx(double x){
    Coordx=x;
}
//����Coordy
void CityInfo::SetCoordy(double y){
    Coordy=y;
}
//�õ���������
string CityInfo::GetName( ){
    return Name;
}
//�õ������±�����
int CityInfo::GetCityIndex( ){
    return CityIndex;
}
//�õ�X����
double CityInfo::GetCoordx( ){
    return Coordx;
}
//�õ�Y����
double CityInfo::GetCoordy(){
    return Coordy;
}
//�õ��������о���
double CityInfo::GetCityDis(CityInfo c1){
	return sqrt((c1.GetCoordx()-Coordx)*(c1.GetCoordx()-Coordx)+(c1.GetCoordy()-Coordy)*(c1.GetCoordy()-Coordy));
}