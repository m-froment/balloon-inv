c ----------------------------------------------------------------------
c  
c          Marouchka Froment
c          NORSAR 
c
c          Code adapted from 
c          Johannes Schweitzer
c          NORSAR
c
c          KJELLER, Norway
c
c  e-mail: johannes.schweitzer@norsar.no
c  e-mail: marouchka.froment@norsar.no 
c
c----------------------------------------------------------------------
c
c----------------------------------------------------------------------
c
c                        Short desciption 
c
c     This program is adapted from the hyposat_loc.f subroutine of the
c     HYPOSAT software of NORSAR 
c          see DOI: 10.2312/GFZ.NMSOP-3_PD_11.1
c
c     developed from laufps.f
c          see DOI: 10.2312/GFZ.NMSOP-2_PD_11.2
c
c
c     Only the travel time calculation was kept 
c     It calculates ray travel times, backazimuths, and slowness values.
c
c     All calculations are done for a spherical planet with arbitrary 
c     radius, with travel times corrected for the ellipticity. 
c
c----------------------------------------------------------------------
c
c               Program History
c
c     October 2024: Creation of a single program TTPLANET
c
c     Oct 2024: Note: when the source is exactly at the interface
c               between two layers, the phase labelling doesn't work
c               for head waves (Sn, Pn) 
c
c     April 2025: discovered a problem in this code with 
c     Low Velocity Zones. Required updates. 
c     FIRST TRUSTED VERSION FOR LVZ -- works
c
c----------------------------------------------------------------------

c----------------------------------------------------------------------
      subroutine ttloc(ho,dis,z,v0,azo,jmod,re,
     +                 locgeo,typctl,
     +                 nphas2,ttc,phcd, dtdd, 
     +                 ierr)

c----------------------------------------------------------------------
c     Here we give informations on the subroutine call

c     ho:      input, source depth (km)
c     dis:     input, receiver distance (degrees)
c     z:       input, array of layer depths (km)
c     v0:      input, array of P and S velocities (2,km/s)
c     azo:     input, array of strings identifying MOHO 
c     jmod:    input, number of layers in model 
c     re:      input, planet radius (km)
c     locgeo:  input, flag for more precise ray tracing 
c     typctl:  input, level of verbosity (8+) 

c     nphas2:  output, number of phases found 
c     ttc:     output, array of arrival times [sec]
c     phcd:    output, array of phase names 
c     dtdd:    output, ray parameters of onsets in [sec/deg]
c     ierr:    output, error flag (0 or else)
c----------------------------------------------------------------------

c      implicit real*8 (a-h,o-z)
c      implicit integer (i-n)
      IMPLICIT NONE 

      REAL*8      ho, dis
      real*8      PI,PIM,AA,del,D,V2,V,zdiff,PAD,G1,G2,G3,P,O,
     +            G, VHQ, VMAX, C, B, VV, E, F,Q,R,RR,PA,TT,
     +            ttp,ppp,FCT1,FCT2,FCT3,TT1,PA1,T ,FA 
      INTEGER     nphas2,ierr,typctl, jmod,ndisc
      integer     i, i2, ib1, ib2, ibm, ibn, icon, ij, imoh, 
     +            ipd, iph, iq5, isd, izo, j, k, kk, 
     +            m, ni, nphas
      logical     locgeo   

c     mphas  = maximum number of phases for tt-calculations 
      integer    mphas
      parameter (mphas = 120)
      character phcd(mphas)*8
      real*8    ttc(mphas),dtdd(mphas) 

      real*8 re,  zflat
      real*8 zmax
      real*8 rmax

c     maxla = maximum number of layers for a velocity model
      integer maxla 
      parameter (maxla = 101)
      CHARACTER phas1*8,az(maxla)*4
      real*8 zo,h(maxla), dum

      real*8   z(maxla), v0(2, maxla), zd(maxla),vd(2,maxla)
      character azo(maxla)*4
      
c     NP    = maximum number of calculated (defined) phases
      integer np 
      parameter (np=50)
      character phlist(np)*8      
      character phase*8

      integer   indx(np), nphas1, phnum, iqq,ion, iz

c----------------------------------------------------------------------
c     Here we pass information to f2py on what is an input or output

cf2py intent(in) ho 
cf2py intent(in) dis
cf2py intent(in) z 
cf2py intent(in) v0 
cf2py intent(in) jmod 
cf2py intent(in) re
cf2py intent(in) locgeo 
cf2py intent(in) typctl 
c
cf2py intent(out) nphas2
cf2py intent(out) ttc 
cf2py intent(out) phcd
cf2py intent(out) dtdd
cf2py intent(out) ierr
c----------------------------------------------------------------------

      DIMENSION RR(2),TT(2),V(2,maxla),G(2,maxla),V2(2,maxla),
     *          ttp(np),ppp(np),ion(np), 
     *          PAD(2),VHQ(2),FA(maxla),pa(2),
     *          ndisc(maxla)
      real*8  tti(np)
      real*8  rtt(np)

      INTEGER IB

      logical   kp, ks

      data phlist/'Pg','Pb','Pn','P','PbP','PmP',
     +            'Sg','Sb','Sn','S','SbS','SmS',38*' '/

      real*8 zold

      save 

c     Flag for max depth
      zmax= 6350.d0
      rmax=180.d0
      PI=4.d0*DATAN(1.d0)
      PIM=PI/180.d0
      AA=PIM*RE

c     Ray density
      IB = 100
c     MF: check if source is close and high ray density needed
      if(locgeo.or.dis.le.0.1d0) IB = 500
      IBN= IB*10

      del = dis

      if(dis.gt.rmax) then
        print *, 'Model not defind beyond distance ',rmax
        stop
      endif

c     Initializes error flag and number of phases found to 0
      ierr = 0
      nphas2 = 0

c
c     reset onset table
c
c     MF: Loop 110 is done to initialize an empty array 

      do 110 i=1,np
      ion(i) = 0
c     MF: Added initialization to zero for ttp and ppp 
      ttp(i) = 0
      ppp(i) = 0
110   continue


      if(ho.gt.zmax) then
         ierr = 99
         print *,'Depth greater than maximum model depth'
         go to 9000
      endif

      zo = ho

C     Now model building and phase generation 
c
      if(dabs(ho-zold).lt.1.d-5 .and. izo.gt.0) goto 812

      imoh = 999
      icon = 999
      ipd  = 999
      isd  = 999

      zold = ho

c
c     Earth flattening approximation 
c
      do 50 iz = 1,jmod
      call efad (Z(iz),V0(1,iz),zd(iz),Vd(1,iz),re)
      call efad (Z(iz),V0(2,iz),dum,Vd(2,iz),re)
50    continue

      m = jmod

      call efad (zo,dum,zflat,dum,re)

      ij = 0
      izo = 0
      IQQ = 0

      DO 500 I=1,m

      ij = ij + 1
      
      h(ij)   = zd(i)
      v(1,ij) = vd(1,i)
      v(2,ij) = vd(2,i)
      az(ij)  = azo(i)

      IF(AZ(IJ).EQ.'CONR')  ICON = IJ
      IF(AZ(IJ).EQ.'MOHO')  IMOH = IJ
      IF(V(1,IJ).GE.10.D0 .and. IPD.eq.999)  IPD  = IJ
      IF(V(2,IJ).GE.5.5D0 .and. ISD.eq.999)  ISD  = IJ

c     Finds if source is at interface      

      if(izo.eq.0) then

         if(dabs(zflat-h(ij)).lt.1.d-4) then
            IQQ = IJ
            izo = 1
            goto 500
         endif

c     Finds if layer contains the source  

         i2  = i + 1
         if(zd(i2).gt.zflat .and. zd(i).lt.zflat) then
            d = (zflat-zd(i))/(zd(i2)-zd(i))
            ij = ij + 1
            IQQ = IJ
            v(1,ij)=(Vd(1,I2)-Vd(1,I))*d + Vd(1,I)
            v(2,ij)=(Vd(2,I2)-Vd(2,I))*d + Vd(2,I)
            az(ij) = ''
            h(ij)  = zflat
            izo = 1

            IF(V(1,IJ).GE.10.D0 .and. IPD.eq.999)  IPD  = IJ
            IF(V(2,IJ).GE.5.5D0 .and. ISD.eq.999)  ISD  = IJ
         endif

      endif

500   continue
      j = IJ
      m = j - 1 

c     We will at first calculate P than S phases
C     (k-loop)
C
c     MF: Loop for K=1 (P waves) and K=2 (S waves)

      DO 810 K=1,2

      DO 800 I=1,M

      I2=I+1
      V2(K,I)=V(K,I2)

      IF(dabs(V2(K,I)-V(K,I)).le.0.001d0) THEN
         V2(K,I)=1.000001d0*V(K,I)
         V(K,I2)=V2(K,I)
      ENDIF

      zdiff=H(I2)-H(I)
      ndisc(i) = 0
      IF(dabs(zdiff).le.0.0001d0)  then
         zdiff = 1.d-6*H(i)
         H(i2)= H(i)+zdiff
         ndisc(i) = 1
      endif

      G(K,I)=(V2(K,I)-V(K,I))/zdiff

800   continue
      
      if(typctl.gt.10) then
        print *,'Model used:'
        print *,'i z h v(1) v(2) v2(1) v2(2) g(1) g(2)'
        do 811 i=1,j
        print*,i,z(i),h(i),az(i),v(1,i),v(2,i),v2(1,i),v2(2,i),
     +  g(1,i),g(2,i), ndisc(i)
811      continue
      endif

810   continue

812   continue

c
c     here model with equal layering for P and S velocities is read
c     now travel time calculations follow
c

c     MF: New loop on K=1 (P waves) or K=2 (S waves)

      DO 7500 K=1,2

      if(k.eq.1) then
        kp=.true.
        ks=.false.
      else
        kp=.false.
        ks=.true.
      endif

      VHQ(K)=V(K,IQQ)*V(K,IQQ)
      PAD(K)=(RE-H(IQQ))*PIM/V(K,IQQ)

      IF(IQQ.EQ.1)  GO TO 1000

C
C     direct waves ( if source deeper than 0.)
c

      if(kp) then
         phase='Pg      '
         if(iqq.gt.icon) phase='Pb      '
         if(iqq.gt.imoh) phase='Pn      '
         if(iqq.ge.ipd)  phase='P       '
      else if(ks) then
         phase='Sg      '
         if(iqq.gt.icon) phase='Sb      '
         if(iqq.gt.imoh) phase='Sn      '
         if(iqq.ge.isd)  phase='S       '
      endif

      CALL REFLEX(IQQ,K,rr,tt,pa,fa,ttp,ppp,aa,del,
     +                VHQ,PAD,PI,V,G,V2,ndisc,ion,
     +                IB,IQQ,phase)

C
C     body waves 
C  

1000  continue

      pa(1) = 0.d0
      pa(2) = 0.d0
      rr(1) = 0.d0
      rr(2) = 0.d0
      tt(1) = 0.d0
      tt(2) = 0.d0

      IQ5=1
      IF(IQQ.GT.1)  IQ5=IQQ
      VMAX=V(K,IQ5)

      DO 1300 I=1,IQ5
      FA(I)=1.d0
      IF(VMAX.LT.V(K,I)) VMAX=V(K,I)
1300  continue

      DO 3000 I=IQ5,M

      FA(I)=2.d0

      D=V2(K,I)
      IF(D.LE.VMAX)  GO TO    3000

      if (ndisc(i).ne.0) go to 2999

      ib2 = ib
      ib1 = 1
      ibm = 0

      FA(I)=2.d0

      if(kp) then
         phase='Pg      '
         if(i.eq.icon) phase='PbP     '
         if(i.gt.icon) phase='Pb      '
         if(i.eq.imoh) phase='PmP     '
         if(i.gt.imoh) phase='Pn      '
         if(i.gt.ipd)  phase='P       '
      else if(ks) then
         phase='Sg      '
         if(i.eq.icon) phase='SbS     '
         if(i.gt.icon) phase='Sb      '
         if(i.eq.imoh) phase='SmS     '
         if(i.gt.imoh) phase='Sn      '
         if(i.gt.isd)  phase='S       '
      endif

      C=V(K,I)
      IF(C.LT.VMAX) C=VMAX

1350  G1=DBLE(IB2-1)
      B=(D-C)/G1

      DO 2500 I2=ib1,IB2

      G2=dble(i2-1)
      VV=C+G2*B
      R=0.D0
      T=0.D0
C
      DO 2000 KK=1,I

      if(ndisc(kk).ne.0) go to 2000

      E=V(K,KK)
      G1=E/VV
      P=DSQRT(DABS(1.D0-G1*G1))
      O=1.d0/G(K,KK)

      IF(KK.lt.I)  THEN
         F=V2(K,KK)
         G3=F/VV
         Q=DSQRT(DABS(1.D0-G3*G3))
      else
         F=VV
         Q=0.d0
      ENDIF

      R=R+FA(KK)*(P-Q)*O
      T=T+FA(KK)*DLOG(F*(1.D0+P)/(E*(1.D0+Q)))*O

2000  CONTINUE
c

      RR(2) = R*VV/AA
      TT(2) = T
      PA(2) = AA/VV

      phas1 = phase

      iph=phnum(phas1)

      IF (RR(2).EQ.del) THEN

         ion(iph) = ion(iph)+1

         if(ion(iph).eq.1) then
            ttp(iph)=TT(2)
            ppp(iph)=PA(2)
         else
            if(TT(2).lt.ttp(iph)) then
               ttp(iph)=TT(2)
               ppp(iph)=PA(2)
            endif
         endif
         GO TO 2400
      ENDIF

      if(i2.le.1) go to 2400

      FCT1=DEL-RR(1)
      FCT2=DEL-RR(2)

      IF(FCT1*FCT2.LT.0.d0) THEN

         if(ibm .eq. 0 ) then
            ib2 = ibn
            ib1 = (i2-1)*10
            ibm = 1
            go to 1350
         endif
            
         FCT3=FCT1/(RR(2)-RR(1))
         TT1=FCT3*(TT(2)-TT(1))+TT(1)
         PA1=FCT3*(PA(2)-PA(1))+PA(1)

         ion(iph) = ion(iph)+1

         if(ion(iph).eq.1) then
            ttp(iph)=TT1
            ppp(iph)=PA1
         else
            if(TT1.lt.ttp(iph)) then
               ttp(iph)=TT1
               ppp(iph)=PA1
            endif
         endif
      ENDIF

2400  continue

      rr(1) = rr(2)
      tt(1) = tt(2)
      pa(1) = pa(2)

2500  continue
C

2999  VMAX=D
3000  CONTINUE
C
C     End of the body-phase and direct-wave loop
C

7500  continue
      
c
c     Finally we have to do some interpolations
c

      nphas1 = 0

      do 8800 i=1,np
    
      if (ion(i).eq.0) go to 8500

      nphas1= nphas1 + 1

      tti(nphas1) = ttp(i)
      dtdd(nphas1) = ppp(i)
      phcd(nphas1) = phlist(i)

8500  continue

      if(typctl.gt.8 .and. nphas1.gt.0 .and. ni.gt.0) then
         print *,'[loc]', i,nphas,nphas1,phcd(nphas),
     *        tti(nphas1),dtdd(nphas)
      endif

8800  continue

      do 8850 i = 1,nphas1
        rtt(i) = tti(i)
8850  continue

      call indexx(nphas1,rtt,indx)

      do 8900 i=1,nphas1

        j       = indx(i)
  
        ttc(i)  = tti(j)
        phcd(i) = phcd(j)
        dtdd(i) = dtdd(j)

c       The final print
        if(typctl.ge.8) then
           print *,i,j,dis,phcd(i),ttc(i),dtdd(i)
        endif

8900  continue

      nphas2 = nphas1


9000  RETURN
      END
C

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE  REFLEX(II,K,rr,tt,pa,fa,ttp,ppp,aa,del,
     +                    VHQ,PAD,PI,V,G,V2,ndisc,ion,
     +                    IB,IQQ,phase)
      IMPLICIT NONE 
c     IMPLICIT real*8 (A-H,O-Z)
C
      integer maxla, np , ndisc,ion,IB,IQQ
      integer I,IBC, IPH, KK, L 
      PARAMETER (maxla=101,np=50)

      real*8 rr,tt,pa,fa,ttp,ppp,aa,del,VHQ,PAD,PI,V,G,V2
      real*8 B,DT,E,F,FCT1,FCT2,FCT3,FI, G1, O,P,PA1,
     +       Q,R,RVV,T,TT1,VMAX 

      DIMENSION RR(2),TT(2),V(2,maxla),G(2,maxla),V2(2,maxla),
     *          ttp(np),ppp(np),ion(np),
     *          PAD(2),VHQ(2),FA(maxla),pa(2),
     *          ndisc(maxla)

      character phase*8


      integer phnum,ii,k

c      MF: Attempt at removing SAVE statement 
c      As it can break reproducibility in Python 
c      SAVE

      iph=phnum(phase)
      L=II-1

c      MF
c      print *, ii, k 

         VMAX=V(K,IQQ)
c     MF
c      print *, VMAX

         DO  1000  I=1,L
         FA(I)=2.D0
         IF(I.LT.IQQ)  FA(I)=FA(I)-1.D0
         IF(V(K,I) .GT.VMAX)  VMAX=V(K,I)
         IF(V2(K,I).GT.VMAX)  VMAX=V2(K,I)
c     MF
c      print *, VMAX 
1000     CONTINUE

      IBC=IB*30
      IF(IBC.GT.1200) IBC=1200

      RR(1)=0.d0
      RR(2)=0.d0
      PA(1)=0.d0
      PA(2)=0.d0
      TT(1)=0.d0
      TT(2)=0.d0

      B=PI/(2.d0*DBLE(IBC-1))

      DO  1500 I=1,IBC
      RVV=DSIN(B*DBLE(I-1))/VMAX
      T=0.D0
      R=0.D0

      DO  1101  KK=1,L
      if(FA(KK).lt.0.9d0 .or. ndisc(kk).ne.0) go to 1101
      E=V(K,KK)
      G1=E*RVV
      P=DSQRT(DABS(1.D0-G1*G1))
      O=1.d0/G(K,KK)
      F=V2(K,KK)
      G1=F*RVV
      Q=DSQRT(DABS(1.D0-G1*G1))
      R=R+FA(KK)*(P-Q)*O
      DT=FA(KK)*DLOG(F*(1.D0+P)/(E*(1.D0+Q)))*O
      T=T+DT
1101  CONTINUE

      TT(2)=T

      IF(I.GT.1)  then
         RR(2)=R/(RVV*AA)
         P=DSQRT(DABS(1.D0/(RVV*RVV*VHQ(K))-1.D0))
         IF(P.LE.0.D0)  then
           FI=0.5d0*PI
         else
           FI=DATAN(1.D0/P)
         endif
      else
         FI    = 0.0D0
         rr(2) = 0.0d0
      endif

      IF(II.EQ.IQQ)  FI=pi-FI

      PA(2)=DSIN(FI)*PAD(K)
      if(pa(2).lt.1.d-4) pa(2)=0.0d0

      IF (dabs(RR(2)-del).le.1.d-5) THEN

         ion(iph) = ion(iph)+1

         if(ion(iph).eq.1) then
            ttp(iph)=TT(2)
            ppp(iph)=PA(2)
         else
            if(TT(2).lt.ttp(iph)) then
               ttp(iph)=TT(2)
               ppp(iph)=PA(2)
            endif
         endif
         GO TO 1400
      ENDIF

      if(i.eq.1) go to 1400

      FCT1=del-RR(1)
      FCT2=del-RR(2)

      IF(FCT1*FCT2.LT.0.d0) THEN
         FCT3=FCT1/(RR(2)-RR(1))
         TT1=FCT3*(TT(2)-TT(1))+TT(1)
         PA1=FCT3*(PA(2)-PA(1))+PA(1)

         ion(iph) = ion(iph)+1

         if(ion(iph).eq.1) then
            ttp(iph)=TT1
            ppp(iph)=PA1
         else
            if(TT1.lt.ttp(iph)) then
               ttp(iph)=TT1
               ppp(iph)=PA1
            endif
         endif
      ENDIF

1400  CONTINUE

      tt(1) = tt(2)
      rr(1) = rr(2)
      pa(1) = pa(2)
 
1500  CONTINUE

      RETURN
      END
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      function phnum(phase)

c      include 'phlist.h'
c     MF: Replaced by copy of file phlist.h 
c     if changing np, change also np in subroutine reflex(1)
c
      parameter (np=50)

      character phlist(np)*8
      data phlist/'Pg','Pb','Pn','P','PbP','PmP',
     +            'Sg','Sb','Sn','S','SbS','SmS',38*' '/

c     ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc      

      CHARACTER phase*8

      integer phnum

c      MF: Attempt at removing SAVE statement 
c      As it can break reproducibility in Python 
c      SAVE

      phnum = 999
      do 5 i = 1,np
      if(phase.eq.phlist(i)) then
        phnum = i
        return
      endif
5     continue
      return
      end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c     Subroutines apended below: indexx, efad 

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine efad(zsp,vsp,zfl,vfl, re)
C
c     input : depth and velocity (zsp, vsp) in a spherical Earth model
c
c     output: depth and velocity (zfl, vfl) in an equivalent flat 
c             Earth model
c
C     author: Johannes Schweitzer, NORSAR
C             June 2004
C
c     MF: Note: For f2py, subroutine arguments cannot be implicit 
C       IMPLICIT real*8 (A-H,O-Z)

      real*8 zfl,vfl, zsp,vsp, f, re

      if(zsp.lt.re) then
         f = re/(re-zsp)
         vfl = vsp*f
         zfl = re*dlog(f)
C          print *, 'RES EFAD', vfl, zfl 
      else
        print *,'Earth-Flattening Approximation is not defined'
        print *,'for depths equal or larger than 6371 km !!! '
        print *,' Check model input!!!'
        stop
      endif

      return
      end
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      SUBROUTINE indexx(n,arr,indx)
C  (C) Copr. 1986-92 Numerical Recipes Software
      INTEGER n,indx(n),M,NSTACK
      REAL*8 arr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,indxt,ir,itemp,j,jstack,k,l,istack(NSTACK)
      REAL*8 a
      do 11 j=1,n
        indx(j)=j
11    continue
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 13 j=l+1,ir
          indxt=indx(j)
          a=arr(indxt)
          do 12 i=j-1,1,-1
            if(arr(indx(i)).le.a)goto 2
            indx(i+1)=indx(i)
12        continue
          i=0
2         indx(i+1)=indxt
13      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        itemp=indx(k)
        indx(k)=indx(l+1)
        indx(l+1)=itemp
        if(arr(indx(l+1)).gt.arr(indx(ir)))then
          itemp=indx(l+1)
          indx(l+1)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l)).gt.arr(indx(ir)))then
          itemp=indx(l)
          indx(l)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l+1)).gt.arr(indx(l)))then
          itemp=indx(l+1)
          indx(l+1)=indx(l)
          indx(l)=itemp
        endif
        i=l+1
        j=ir
        indxt=indx(l)
        a=arr(indxt)
3       continue
          i=i+1
        if(arr(indx(i)).lt.a)goto 3
4       continue
          j=j-1
        if(arr(indx(j)).gt.a)goto 4
        if(j.lt.i)goto 5
        itemp=indx(i)
        indx(i)=indx(j)
        indx(j)=itemp
        goto 3
5       indx(l)=indx(j)
        indx(j)=indxt
        jstack=jstack+2
c j.s.  if(jstack.gt.NSTACK)pause 'NSTACK too small in indexx'
        if(jstack.gt.NSTACK) stop  'NSTACK too small in indexx'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


c
c     input:
c 
c             ho      = source depth in km 
c
c             dis     = receiver distance in deg
c
c             typctl  = verbosity level
c
c    in common MODEL :
c 
c             v0(1,i) =  P velocity in layer i
c             v0(2,i) =  S velocity in layer i
c
c             z(i)    =  depth of layer i
c
c             elatc   =  latitude  to get the right CRUST 1.0 model 
c                        parameters
c             elonc   =  longitude to get the right CRUST 1.0 model 
c                        parameters
c
c             elat2   =  latitude  to get the right CRUST 1.0 model 
c                        parameters at second modelled point
c             elon2   =  longitude to get the right CRUST 1.0 model 
c                        parameters at second modelled point
c
c             imo     <= 0
c                     <= 2 reading of model from filloc
c                 (   =  2 CRUST 1.0 only used for station
c                          corrections  )
c                     =  3 using model from CRUST 1.0
c                     =  4 dito + CRUST 1.0 used for travel-time 
c                          corrections
c
c             jmod    =  number of layers
c
c             filloc  =  filename for file with local velocity model
c
c             mtyp    =  CRUST 1.0 model type
c
c             locgeo  =  false/true : if 'true', higher density of
c                        'rays' (IB) and smaller distances are used (DIS)
c         common
c     output 
c
c             nphas2  = number of found onsets at receiver distance
c
c             phcd    = array with names of found onsets
c
c             ttc     = travel times of onsets in [sec]
c
c             dtdd    = ray parameters of onsets in [sec/deg]
c
c             ierr    = 0     everything o.k.
c                       else  some error occurred.
c
c    in common MODEL :
c 
c             v0(1,i) =  P velocity in layer i
c             v0(2,i) =  S velocity in layer i
c
c             z(i)    =  depth of layer i
c
c             jmod    =  number of layers
c
c             azo(i)  =  Conrad/Moho indicator
c
c             rmax    =  maximum distance for which this model shall be
c                        used (read in from model file).
c
c             zmax    =  maximum depth for which this model can be
c                        used (read in from model file).
c
c
c     INPUT PARAMETER TYPE 
