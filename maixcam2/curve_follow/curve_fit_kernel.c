#include <math.h>
#include <stddef.h>

/* Identical three-iteration local quadratic IRLS normal equations.
 * Coordinates and arc lengths are supplied by the Python reference path.
 * No control, threshold selection, extrapolation or tracking runs here. */
int curve_fit_local(const double *points, const double *arc, int n,
                    double *fitted, double *residual) {
    if (n < 5 || n > 160) return 0;
    for (int i=0; i<n; ++i) {
        double u[160], base[160], weight[160];
        int count=0;
        for (int j=0; j<n; ++j) {
            u[j]=(arc[j]-arc[i])/.12;
            base[j]=fabs(u[j])<=1. ? exp(-u[j]*u[j]) : 0.;
            weight[j]=base[j];
            count += base[j]>0.;
        }
        /* Unusual sparse input uses the Python reference's nearest support. */
        if (count<5) return 0;
        double coefficients[3][2];
        for (int iteration=0; iteration<3; ++iteration) {
            double a[3][5]={{0}};
            for (int j=0; j<n; ++j) {
                if (base[j]==0.) continue;
                double design[3]={1.,u[j],u[j]*u[j]};
                for (int row=0; row<3; ++row) {
                    double w=weight[j]*design[row];
                    for (int col=0; col<3; ++col) a[row][col]+=w*design[col];
                    a[row][3]+=w*points[2*j];
                    a[row][4]+=w*points[2*j+1];
                }
            }
            for (int row=0; row<3; ++row) a[row][row]+=1e-10;
            for (int col=0; col<3; ++col) {
                int pivot=col;
                for (int row=col+1; row<3; ++row)
                    if (fabs(a[row][col])>fabs(a[pivot][col])) pivot=row;
                if (!isfinite(a[pivot][col]) || fabs(a[pivot][col])<1e-30) return 0;
                if (pivot!=col) for (int k=col; k<5; ++k) {
                    double swap=a[col][k]; a[col][k]=a[pivot][k]; a[pivot][k]=swap;
                }
                for (int row=col+1; row<3; ++row) {
                    double factor=a[row][col]/a[col][col];
                    for (int k=col; k<5; ++k) a[row][k]-=factor*a[col][k];
                }
            }
            for (int dim=0; dim<2; ++dim) for (int row=2; row>=0; --row) {
                double value=a[row][3+dim];
                for (int col=row+1; col<3; ++col) value-=a[row][col]*coefficients[col][dim];
                coefficients[row][dim]=value/a[row][row];
            }
            for (int j=0; j<n; ++j) {
                if (base[j]==0.) continue;
                double x=coefficients[0][0]+u[j]*coefficients[1][0]+u[j]*u[j]*coefficients[2][0];
                double y=coefficients[0][1]+u[j]*coefficients[1][1]+u[j]*u[j]*coefficients[2][1];
                double error=hypot(x-points[2*j],y-points[2*j+1]);
                weight[j]=base[j]*fmin(1.,.008/fmax(error,1e-6));
            }
        }
        fitted[2*i]=coefficients[0][0];fitted[2*i+1]=coefficients[0][1];
        residual[i]=hypot(fitted[2*i]-points[2*i],fitted[2*i+1]-points[2*i+1]);
        if (!isfinite(residual[i])) return 0;
    }
    return 1;
}

#include <stdlib.h>
static int compare_double(const void *aa,const void *bb) {
    double a=*(const double*)aa,b=*(const double*)bb;
    return (a>b)-(a<b);
}
static double local_quantile(double *values,int n,double fraction) {
    qsort(values,n,sizeof(double),compare_double);
    double position=(n-1)*fraction;int low=(int)position;
    int high=low+1<n?low+1:low;double w=position-low;
    double difference=values[high]-values[low];
    return w<.5?values[low]+difference*w:values[high]-difference*(1.-w);
}
/* Same bidirectional overlap/tangent association as geometry.curve_agreement.
 * 0: unsupported input (Python fallback), 1: result, 2: no valid overlap.
 * No observation age, identity, control, or new association policy lives here. */
int curve_agree_local(const double *a,const double *b,const double *ta,const double *tb,
                      const double *sa,const double *sb,int na,int nb,double *out) {
    if (na<5 || nb<5 || na>160 || nb>160) return 0;
    double da[160],db[160];int ja[160],jb[160];
    for(int i=0;i<na;i++){da[i]=INFINITY;ja[i]=0;}
    for(int j=0;j<nb;j++){db[j]=INFINITY;jb[j]=0;}
    for(int i=0;i<na;i++) for(int j=0;j<nb;j++) {
        double x=a[2*i]-b[2*j],y=a[2*i+1]-b[2*j+1],d=x*x+y*y;
        if(d<da[i]){da[i]=d;ja[i]=j;}
        if(d<db[j]){db[j]=d;jb[j]=i;}
    }
    out[0]=0.;out[1]=INFINITY;out[2]=INFINITY;
    for(int direction=0;direction<2;direction++) {
        int n=direction?nb:na,other=direction?na:nb;
        const int *indices=direction?jb:ja;
        const double *distance=direction?db:da,*support=direction?sb:sa;
        const double *t1=direction?tb:ta,*t2=direction?ta:tb;
        double distances[160],headings[160];int count=0,first=-1,last=-1;
        for(int i=0;i<n;i++) {
            int j=indices[i];if(j<=0 || j>=other-1)continue;
            if(first<0)first=i;last=i;
            distances[count]=sqrt(distance[i]);
            headings[count]=t1[2*i]*t2[2*j]+t1[2*i+1]*t2[2*j+1];count++;
        }
        if(count<5 || support[last]-support[first]<.06)return 2;
        double d=local_quantile(distances,count,.8),h=local_quantile(headings,count,.2);
        out[0]=fmax(out[0],d);out[1]=fmin(out[1],h);
        out[2]=fmin(out[2],support[last]-support[first]);
    }
    return 1;
}

/* Same filtering/arc interpolation and derivative formula as geometry.py.
 * Return 0 only for the Python fallback, -1 for an invalid observed chain. */
static int resample_chain(const double *p,int n,double *out) {
    if(n<3 || n>1600)return -1;
    double clean[3200],s[1600];int count=0;
    for(int i=0;i<n;i++) {
        if(!isfinite(p[2*i]) || !isfinite(p[2*i+1]))return -1;
        if(i && hypot(p[2*i]-p[2*i-2],p[2*i+1]-p[2*i-1])<=1e-5)continue;
        clean[2*count]=p[2*i];clean[2*count+1]=p[2*i+1];count++;
    }
    if(count<3)return -1;
    s[0]=0.;
    for(int i=1;i<count;i++)s[i]=s[i-1]+hypot(clean[2*i]-clean[2*i-2],clean[2*i+1]-clean[2*i-1]);
    double length=s[count-1];if(length<.075 || length>2.)return -1;
    int m=(int)(length/.015)+1;if(m<10)m=10;if(m>160)m=160;
    double step=length/(m-1);int j=0;
    for(int i=0;i<m;i++) {
        double q=i==m-1?length:i*step;
        while(j<count-2 && s[j+1]<q)j++;
        double w=(q-s[j])/(s[j+1]-s[j]);
        for(int d=0;d<2;d++)out[2*i+d]=clean[2*j+d]+w*(clean[2*j+2+d]-clean[2*j+d]);
    }
    return m;
}
static void planar_gradient(const double *v,const double *s,int n,double *out) {
    for(int d=0;d<2;d++) {
        out[d]=(v[2+d]-v[d])/(s[1]-s[0]);
        out[2*(n-1)+d]=(v[2*(n-1)+d]-v[2*(n-2)+d])/(s[n-1]-s[n-2]);
    }
    for(int i=1;i<n-1;i++) {
        double before=s[i]-s[i-1],after=s[i+1]-s[i];
        double a=-after/(before*(before+after)),b=(after-before)/(before*after),c=before/(after*(before+after));
        for(int d=0;d<2;d++)out[2*i+d]=a*v[2*(i-1)+d]+b*v[2*i+d]+c*v[2*(i+1)+d];
    }
}
int curve_fit_chain(const double *points,int n,double *out,double *stats) {
    if(n>1600)return 0;
    double sampled[320],s[160],fitted[320],residual[160],final[320];
    int m=resample_chain(points,n,sampled);if(m<0)return -1;
    s[0]=0.;for(int i=1;i<m;i++)s[i]=s[i-1]+hypot(sampled[2*i]-sampled[2*i-2],sampled[2*i+1]-sampled[2*i-1]);
    if(!curve_fit_local(sampled,s,m,fitted,residual))return 0;
    stats[0]=local_quantile(residual,m,.9);stats[1]=local_quantile(residual,m,.5);
    int size=resample_chain(fitted,m,final);if(size<0)return -1;
    s[0]=0.;for(int i=1;i<size;i++)s[i]=s[i-1]+hypot(final[2*i]-final[2*i-2],final[2*i+1]-final[2*i-1]);
    double derivative[320],tangent[320],second[320];
    planar_gradient(final,s,size,derivative);
    for(int i=0;i<size;i++) {
        double norm=fmax(hypot(derivative[2*i],derivative[2*i+1]),1e-8);
        tangent[2*i]=derivative[2*i]/norm;tangent[2*i+1]=derivative[2*i+1]/norm;
    }
    planar_gradient(tangent,s,size,second);stats[2]=s[size-1];stats[3]=INFINITY;
    for(int i=0;i<size;i++) {
        out[5*i]=final[2*i];out[5*i+1]=final[2*i+1];out[5*i+2]=tangent[2*i];out[5*i+3]=tangent[2*i+1];
        out[5*i+4]=tangent[2*i]*second[2*i+1]-tangent[2*i+1]*second[2*i];
        if(i)stats[3]=fmin(stats[3],tangent[2*i]*tangent[2*i-2]+tangent[2*i+1]*tangent[2*i-1]);
    }
    return size;
}

/* Exact point-to-polyline distance and arclength, matching support_projection.
 * No validity, coverage, fitting, target or control decisions run in this loop. */
int curve_project_polyline(const double *q,int nq,const double *r,int nr,double *dist,double *pos) {
    if(nq<1 || nq>1600 || nr<2 || nr>1600)return 0;
    for(int i=0;i<2*nq;i++)if(!isfinite(q[i]))return 0;
    for(int i=0;i<2*nr;i++)if(!isfinite(r[i]))return 0;
    for(int i=0;i<nq;i++){dist[i]=INFINITY;pos[i]=0.;}
    double arc=0.;
    for(int j=0;j<nr-1;j++){
        double dx=r[2*j+2]-r[2*j],dy=r[2*j+3]-r[2*j+1];
        double length2=dx*dx+dy*dy,den=fmax(length2,1e-12);
        for(int i=0;i<nq;i++){
            double t=((q[2*i]-r[2*j])*dx+(q[2*i+1]-r[2*j+1])*dy)/den;
            t=fmax(0.,fmin(1.,t));
            double x=q[2*i]-(r[2*j]+t*dx),y=q[2*i+1]-(r[2*j+1]+t*dy),d=x*x+y*y;
            if(d<dist[i]){dist[i]=d;pos[i]=arc+t*sqrt(den);}
        }
        arc+=sqrt(length2);
    }
    for(int i=0;i<nq;i++)dist[i]=sqrt(dist[i]);
    return 1;
}

/* Fuse the reference contour's normal probes, cyclic chain split and facing
 * checks. All photometric masks and geometry thresholds are caller supplied;
 * accepted chains still pass the same quadratic fit and association in Python.
 * 0 = unavailable/invalid input (reference fallback), >=1 = handled, result+1.
 * Double outputs hold x,y,inward_nx,inward_ny; runs hold offset,count,facing. */
#include <stdint.h>
typedef struct {
    double x,y,nx,ny,norm,road[2],outside[2];
    int visible;
} boundary_point;
static int boundary_sample(const uint8_t *mask,const uint8_t *visible,
                           int rows,int cols,double xs0,double ys0,double step,
                           double x,double y,int *valid) {
    int r=(int)nearbyint((x-xs0)/step),c=(int)nearbyint((ys0-y)/step);
    *valid=r>=0 && r<rows && c>=0 && c<cols;
    if(r<0)r=0;if(r>=rows)r=rows-1;if(c<0)c=0;if(c>=cols)c=cols-1;
    *valid=*valid && visible[r*cols+c];
    return mask[r*cols+c];
}
int curve_boundary_chains(const int32_t *pixels,int n,const uint8_t *codes,
        const uint8_t *gray,const uint8_t *visible,int rows,int cols,
        const double *xs,const double *ys,double step,const double *distances,
        int minimum_samples,double minimum_length,double *out,double *runs) {
    if(n<2 || rows<1 || cols<1 || step<=0 || minimum_samples<2)return 0;
    boundary_point *p=calloc((size_t)n,sizeof(*p));
    uint8_t *good=malloc((size_t)n);
    if(!p || !good){free(p);free(good);return 0;}
    for(int i=0;i<n;i++){
        int c=pixels[2*i],r=pixels[2*i+1];
        if(r<0 || r>=rows || c<0 || c>=cols){free(p);free(good);return 0;}
        p[i].x=xs[r];p[i].y=ys[c];
    }
    for(int i=0;i<n;i++){
        int next=(i+4)%n,prev=(i+n-(4%n))%n;
        double dx=p[next].x-p[prev].x,dy=p[next].y-p[prev].y;
        p[i].norm=sqrt(dx*dx+dy*dy);
        double denominator=fmax(p[i].norm,1e-8);
        p[i].nx=-dy/denominator;p[i].ny=dx/denominator;
        int region[2][3];p[i].visible=1;
        for(int side=0;side<2;side++)for(int d=0;d<3;d++){
            double sign=side ? -1. : 1.;int valid;
            region[side][d]=boundary_sample(codes,visible,rows,cols,xs[0],ys[0],step,
                         p[i].x+sign*distances[d]*p[i].nx,p[i].y+sign*distances[d]*p[i].ny,&valid);
            p[i].visible &= valid;
        }
        for(int side=0;side<2;side++){
            int road=0,exterior=0;
            for(int d=0;d<3;d++){
                int code=region[side][d],opposite=region[1-side][d];
                road+=((code&1)!=0) || ((code&4)!=0 && (opposite&2)!=0);
                exterior+=(code&2)!=0;
            }
            p[i].road[side]=road/3.;p[i].outside[side]=exterior/3.;
        }
    }
    int chains=0,used=0;
    for(int sign=1;sign>=-1;sign-=2){
        int road_index=sign==1?0:1,start=-1;
        for(int i=0;i<n;i++){
            double evidence=p[i].road[0]-p[i].road[1]+.7*(p[i].outside[1]-p[i].outside[0]);
            good[i]=(sign*evidence>.65 && p[i].road[road_index]>=.66 && p[i].visible && p[i].norm>.012);
            if(!good[i] && start<0)start=i;
        }
        if(start<0)continue;
        int begin=-1;
        for(int k=0;k<=n;k++){
            int i=(start+k)%n,on=k<n && good[i];
            if(on && begin<0)begin=k;
            if(on || begin<0)continue;
            int count=k-begin;
            if(count>=minimum_samples){
                int exterior=0;double length=0.;
                for(int j=begin;j<k;j++){
                    int id=(start+j)%n;exterior+=p[id].outside[1-road_index]>0.;
                    if(j>begin){int prev=(start+j-1)%n;length+=hypot(p[id].x-p[prev].x,p[id].y-p[prev].y);}
                }
                if((double)exterior/count>=.5 && length>=minimum_length){
                    int first=(start+begin)%n,last=(start+k-1)%n;
                    int reverse=hypot(p[last].x,p[last].y)<hypot(p[first].x,p[first].y);
                    int checks=0,road=0;
                    for(int j=0;j<count;j++){
                        int id=(start+begin+(reverse?count-1-j:j))%n,valid;
                        int code=boundary_sample(gray,visible,rows,cols,xs[0],ys[0],step,
                               p[id].x+.012*sign*p[id].nx,p[id].y+.012*sign*p[id].ny,&valid);
                        if(valid){checks++;road+=code!=0;}
                    }
                    double facing=checks ? (double)road/checks : 0.;
                    if(facing>=.45){
                        runs[3*chains]=used;runs[3*chains+1]=count;runs[3*chains+2]=facing;chains++;
                        for(int j=0;j<count;j++){
                            int id=(start+begin+(reverse?count-1-j:j))%n;
                            out[4*used]=p[id].x;out[4*used+1]=p[id].y;
                            out[4*used+2]=sign*p[id].nx;out[4*used+3]=sign*p[id].ny;used++;
                        }
                    }
                }
            }
            begin=-1;
        }
    }
    free(p);free(good);return chains+1;
}

/* Same nearest anchor and first supported forward-circle intersection as
 * geometry.py, fused to avoid Python/NumPy dispatch for every future pose. */
int curve_select_pursuit(const double *path,int n,double radius,double forward,
                         double maximum,double *out) {
    if(n<2 || !isfinite(radius) || !isfinite(forward) || forward<=0.)return 0;
    double best=INFINITY,px=0.,py=0.,at=0.,position=0.;int nearest=-1;
    for(int i=0;i<n;i++)if(!isfinite(path[2*i]) || !isfinite(path[2*i+1]))return 2;
    for(int i=0;i<n-1;i++){
        double x=path[2*i],y=path[2*i+1],dx=path[2*i+2]-x,dy=path[2*i+3]-y;
        double l2=dx*dx+dy*dy,length=hypot(dx,dy),lo=0.,hi=1.;
        if(dx>1e-12)lo=fmax(0.,(forward-x)/dx);
        if(dx< -1e-12)hi=fmin(1.,(forward-x)/dx);
        if(l2>1e-12 && lo<=hi && (fabs(dx)>1e-12 || x>=forward)){
            double f=fmin(hi,fmax(lo,-(x*dx+y*dy)/l2));
            double qx=x+f*dx,qy=y+f*dy,d=qx*qx+qy*qy;
            if(d<best){best=d;px=fmax(forward,qx);py=qy;at=position+f*sqrt(l2);nearest=i;}
        }
        position+=length;
    }
    if(nearest<0)return 2;
    double distance=hypot(px,py);
    out[0]=px;out[1]=py;out[2]=at;out[3]=distance;out[4]=0.;out[5]=1.;
    if(radius==0.)return 1;
    if(radius<0. || distance>maximum)return 2;
    if(distance>=radius){out[5]=fmin(1.,radius/distance);return 1;}
    position=at;
    for(int i=nearest+1;i<n;i++){
        double x=path[2*i],y=path[2*i+1],dx=x-px,dy=y-py;int last=0;
        if(x<forward){double f=fmin(1.,fmax(0.,(forward-px)/dx));x=px+f*dx;y=py+f*dy;dx=x-px;dy=y-py;last=1;}
        double length=hypot(dx,dy),a=dx*dx+dy*dy;
        if(a>1e-14){
            double b=px*dx+py*dy,c=px*px+py*py-radius*radius,disc=b*b-a*c;
            if(disc>=-1e-12){
                double f=(-b+sqrt(fmax(0.,disc)))/a;
                if(f>=-1e-10 && f<=1.+1e-10){
                    f=fmin(1.,fmax(0.,f));out[0]=px+f*dx;out[1]=py+f*dy;
                    out[2]=position+f*length;out[3]=hypot(out[0],out[1]);out[4]=1.;return 1;
                }
            }
        }
        position+=length;px=x;py=y;if(last)break;
    }
    distance=hypot(px,py);if(distance<forward || distance>maximum)return 2;
    out[0]=px;out[1]=py;out[2]=position;out[3]=distance;out[4]=2.;out[5]=fmin(1.,distance/radius);return 1;
}

/* Same finite/arc/tangent validation as curve_agreement, before association.
 * Derivatives and arc lengths are computed locally once rather than through
 * many tiny NumPy calls. Policy gates still belong to the caller. */
int curve_agree_paths(const double *a,int na,const double *b,int nb,double *out) {
    if(na<5 || nb<5)return 2;
    if(na>160 || nb>160)return 0;
    double sa[160],sb[160],ta[320],tb[320];
    for(int side=0;side<2;side++){
        const double *p=side?b:a;int n=side?nb:na;
        double *s=side?sb:sa,*t=side?tb:ta;s[0]=0.;
        for(int i=0;i<n;i++){
            if(!isfinite(p[2*i]) || !isfinite(p[2*i+1]))return 2;
            if(i){s[i]=s[i-1]+hypot(p[2*i]-p[2*i-2],p[2*i+1]-p[2*i-1]);if(s[i]-s[i-1]<=1e-6)return 2;}
        }
        planar_gradient(p,s,n,t);
        for(int i=0;i<n;i++){
            double norm=fmax(hypot(t[2*i],t[2*i+1]),1e-8);t[2*i]/=norm;t[2*i+1]/=norm;
        }
    }
    return curve_agree_local(a,b,ta,tb,sa,sb,na,nb,out);
}

/* Nearest sampled vertices, not segment projections. Strict comparison retains
 * the first index on ties, as NumPy argmin does. Policy stays in Python. */
int curve_nearest_samples(const double *query,int nq,const double *reference,int nr,
                          int *indices,double *distances) {
    if(nq<1 || nr<1)return 0;
    for(int i=0;i<nq;i++)
        if(!isfinite(query[2*i]) || !isfinite(query[2*i+1]))return 0;
    for(int j=0;j<nr;j++)
        if(!isfinite(reference[2*j]) || !isfinite(reference[2*j+1]))return 0;
    for(int i=0;i<nq;i++){
        double best=INFINITY;int index=0;
        for(int j=0;j<nr;j++){
            double distance=hypot(query[2*i]-reference[2*j],query[2*i+1]-reference[2*j+1]);
            if(distance<best){best=distance;index=j;}
        }
        indices[i]=index;distances[i]=best;
    }
    return 1;
}

/* Same float32 operation order as NumPy: preserve contrast rounding at mask
 * thresholds. Keep contraction disabled instead of allowing fused multiply-add.
 * Bitwise predicates avoid short-circuit branches and permit SIMD over pixels.
 * Input moments come from finite uint8 HSV planes blurred by OpenCV. */
#pragma STDC FP_CONTRACT OFF
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
int curve_road_regions(const uint8_t *h, const uint8_t *s, const uint8_t *v,
        const uint8_t *visible, const float *mean, const float *second,
        int n, uint8_t *out) {
    if (n < 1) return 0;
    size_t count = (size_t)n;
    for (int i = 0; i < n; i++) {
        float square = mean[i] * mean[i];
        float variance = second[i] - square;
        float deviation = sqrtf(variance > 0.f ? variance : 0.f);
        float delta = (float)v[i] - mean[i];
        float scaled = 60.f * delta;
        float ratio = scaled / (deviation > 1.f ? deviation : 1.f);
        float contrast = 180.f + ratio;
        uint8_t cv = v[i] < 35 ? 0 : (uint8_t)(
            contrast < 0.f ? 0.f : (contrast > 255.f ? 255.f : contrast)
        );
        int seen = visible[i] != 0;
        int green = (h[i] >= 32) & (h[i] <= 95) & (s[i] > 55)
                  & (v[i] > 25) & seen;
        out[i] = (uint8_t)green;
        out[count + i] = cv;
        out[2*count + i] = (uint8_t)(
            (s[i] < 110) & (v[i] >= 35) & (v[i] < 210) & seen & !green
        );
        out[3*count + i] = (uint8_t)(
            (s[i] < 110) & (cv >= 35) & (cv < 210) & seen & !green
        );
    }
    return 1;
}
