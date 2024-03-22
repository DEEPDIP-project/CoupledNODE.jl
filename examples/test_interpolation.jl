
n = 75
# it receives a tuple as input and it outputs 
dw_v = Chain(
        uv -> let v = uv[2]
            # extract only the v component
            println("v: ", size(v))
            v
        end,
        Upsample(:trilinear,size=(2*n,2*n)),
        MeanPool((2,2)),
    )
uv_dw = Chain(SkipConnection(dw_v, (v_dw, uv) -> let u = uv[1]
            # make u and v linear
            u = reshape(u, 100*100, size(u,3))
            v = reshape(v_dw, n*n, size(u,3))
            # and concatenate
            vcat(u,v)
        end),)
# the upscaling pipeline takes as input the linearized uv and outputs a tuple (u,v) on the same grid as u
up_v = Chain(
        uv-> let v=uv[100*100+1:end, :]
            # reshape v
            v = reshape(v, 75,75, size(v,2),1)
            println("v: ", size(v))
            v
        end,
        Upsample(:trilinear,size=(100,100)),
)
uv_up = Chain(SkipConnection(up_v, (v_up, uv) -> let u = uv[1:100*100, :]
            # make u on the grid
            u = reshape(u, 100,100, size(u,2),1)
            println("u: ", size(u))
            println("v_up: ", size(v_up))
            # and return the tuple
            (u,v_up)
        end),)


uv0
y0 = reshape(u_target[:,:,5,end],100,100,1,1)
x0 = reshape(u_target[:,:,1,end],100,100,1,1)
tup_in = (x0,y0)
z, stz = Lux.setup(rng,uv_dw)
uvd, _ = Lux.apply(uv_dw, tup_in, z, stz)
uvd

ud = uvd[1:grid.nux*grid.nuy, :]
vd = uvd[grid.nux*grid.nuy+1:end, :]
vd = reshape(vd, 75,75, size(vd,2))
heatmap(vd[:,:,1],axis=false,cbar=false,aspectratio=1)

z, stz = Lux.setup(rng,uv_up)
uvup, _ = Lux.apply(uv_up, uvd, z, stz)
uup = uvup[1]
vup = uvup[2]
vup = reshape(vup, 100,100, size(vup,2))
heatmap(vup[:,:,1],axis=false,cbar=false,aspectratio=1)

# *****
# test if Julia does what you like in upsampling
n = 75
y0 = reshape(u_target[:,:,5,end],100,100,1,1)
dw = Chain(
        Lux.Upsample(:trilinear,size=(2*n,2*n)),
        Lux.MeanPool((2,2)),
    )
z, stz = Lux.setup(rng,dw)
yd, _ = Lux.apply(dw, y0, z, stz)
up = Lux.Upsample(:trilinear,size=(100,100))
z, stz = Lux.setup(rng,up)
yr, _ = Lux.apply(up, yd, z, stz)
p1=heatmap(y0[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p2=heatmap(yd[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p3=heatmap(yr[:,:,1,1],axis=false,cbar=false,aspectratio=1)
e1=(y0.-yr)[:,:,1,1]
p4=heatmap(e1,axis=false,cbar=false,aspectratio=1)
plot(p1,p2,p3,p4,layout=(1,4))

yd2 = reshape(imresize(y0[:,:,1,1], (n,n)), n,n,1,1)
yr2 = reshape(imresize(yd2[:,:,1,1], (100,100)), 100,100,1,1)
yd3 = imresize(y0, (n,n), method=Lanczos4OpenCV())
yr3 = imresize(yd3, (100,100), method=Lanczos4OpenCV())
p5=heatmap(y0[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p6=heatmap(yd2[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p7=heatmap(yr2[:,:,1,1],axis=false,cbar=false,aspectratio=1)
e2=(y0.-yr2)[:,:,1,1]
p8=heatmap(e2,axis=false,cbar=false,aspectratio=1)
p9=heatmap(y0[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p10=heatmap(yd3[:,:,1,1],axis=false,cbar=false,aspectratio=1)
p11=heatmap(yr3[:,:,1,1],axis=false,cbar=false,aspectratio=1)
e3=(y0.-yr3)[:,:,1,1]
p12=heatmap(e3,axis=false,cbar=false,aspectratio=1)
plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,layout=(3,4))
sum(e1)
sum(e2)
sum(e3)
