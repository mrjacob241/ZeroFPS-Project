struct Parameters {
    operation: u32, variant: u32, connected: u32, point_count: u32,
    values: vec4<f32>,
    points: array<vec4<f32>, 32>,
}
@group(0) @binding(0) var a_tex: texture_2d<f32>;
@group(0) @binding(1) var b_tex: texture_2d<f32>;
@group(0) @binding(2) var c_tex: texture_2d<f32>;
@group(0) @binding(3) var d_tex: texture_2d<f32>;
@group(0) @binding(4) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var<uniform> p: Parameters;

fn coord(pixel: vec2<u32>, out_size: vec2<u32>, size: vec2<u32>) -> vec2<i32> {
    return vec2<i32>(min(pixel * size / out_size, size - vec2<u32>(1)));
}
fn load_a(q: vec2<u32>, s: vec2<u32>) -> vec4<f32> { return textureLoad(a_tex, coord(q,s,textureDimensions(a_tex)),0); }
fn load_b(q: vec2<u32>, s: vec2<u32>) -> vec4<f32> { return textureLoad(b_tex, coord(q,s,textureDimensions(b_tex)),0); }
fn load_c(q: vec2<u32>, s: vec2<u32>) -> vec4<f32> { return textureLoad(c_tex, coord(q,s,textureDimensions(c_tex)),0); }
fn load_d(q: vec2<u32>, s: vec2<u32>) -> vec4<f32> { return textureLoad(d_tex, coord(q,s,textureDimensions(d_tex)),0); }
fn math(a:f32,b:f32,op:u32)->f32 {
    switch op {
        case 0u:{return a+b;} case 1u:{return a-b;} case 2u:{return a*b;}
        case 3u:{return select(a/b,0.0,abs(b)<=0.000001);}
        case 4u:{return pow(max(a,0.0),b);} case 5u:{return min(a,b);}
        case 6u:{return max(a,b);} default:{return abs(a-b);}
    }
}
fn curve(v:f32)->f32 {
    if p.point_count==0u{return v;} if p.point_count==1u{return p.points[0].y;}
    if p.variant==1u && p.point_count>=4u {
        var low=0.0;var high=1.0;
        for(var step=0u;step<16u;step++){
            let candidate=(low+high)*0.5;let inverse=1.0-candidate;
            let x=inverse*inverse*inverse*p.points[0].x+3.0*inverse*inverse*candidate*p.points[1].x+3.0*inverse*candidate*candidate*p.points[2].x+candidate*candidate*candidate*p.points[3].x;
            if x<v{low=candidate;}else{high=candidate;}
        }
        let t=(low+high)*0.5; let u=1.0-t;
        return u*u*u*p.points[0].y+3.0*u*u*t*p.points[1].y+3.0*u*t*t*p.points[2].y+t*t*t*p.points[3].y;
    }
    var previous=p.points[0].xy;
    for(var i=1u;i<8u;i++){if i>=p.point_count{break;} let next=p.points[i].xy;
        if v<=next.x{return mix(previous.y,next.y,clamp((v-previous.x)/max(next.x-previous.x,0.000001),0.0,1.0));}
        previous=next;
    } return previous.y;
}
fn local(q:vec2<i32>)->vec4<f32>{
    let size=vec2<i32>(textureDimensions(a_tex));
    return textureLoad(a_tex,clamp(q,vec2<i32>(0),size-vec2<i32>(1)),0);
}
fn median_channel(center:vec2<i32>,channel:u32)->f32{
    // Median is deliberately capped to a 5x5 footprint. Larger radii have
    // quadratic storage/sort cost and should use a future histogram kernel.
    let radius=clamp(i32(round(p.values.x)),0,2);
    var values:array<f32,25>;var count=0u;
    for(var y=-2;y<=2;y++){if abs(y)>radius{continue;}for(var x=-2;x<=2;x++){if abs(x)>radius{continue;}
        let value=local(center+vec2<i32>(x,y))[channel];var position=count;
        while position>0u && values[position-1u]>value{values[position]=values[position-1u];position-=1u;}
        values[position]=value;count+=1u;
    }}
    return values[count/2u];
}
fn apply_filter(pixel:vec2<u32>,out_size:vec2<u32>)->vec4<f32>{
    let center=coord(pixel,out_size,textureDimensions(a_tex));
    let radius=clamp(i32(round(p.values.x)),0,16);
    if radius==0{return local(center);}
    if p.variant==2u {
        let center_value=local(center);
        let rgb=5.0*center_value.rgb-local(center+vec2<i32>(1,0)).rgb-local(center-vec2<i32>(1,0)).rgb-local(center+vec2<i32>(0,1)).rgb-local(center-vec2<i32>(0,1)).rgb;
        return vec4<f32>(rgb,center_value.a);
    }
    if p.variant==3u {
        let lum=vec3<f32>(0.2126,0.7152,0.0722);
        let tl=dot(local(center+vec2<i32>(-1,-1)).rgb,lum); let tc=dot(local(center+vec2<i32>(0,-1)).rgb,lum); let tr=dot(local(center+vec2<i32>(1,-1)).rgb,lum);
        let ml=dot(local(center+vec2<i32>(-1,0)).rgb,lum); let mr=dot(local(center+vec2<i32>(1,0)).rgb,lum);
        let bl=dot(local(center+vec2<i32>(-1,1)).rgb,lum); let bc=dot(local(center+vec2<i32>(0,1)).rgb,lum); let br=dot(local(center+vec2<i32>(1,1)).rgb,lum);
        let edge=length(vec2<f32>(-tl+tr-2.0*ml+2.0*mr-bl+br,-tl-2.0*tc-tr+bl+2.0*bc+br));
        return vec4<f32>(edge,edge,edge,local(center).a);
    }
    if p.variant==4u {
        return vec4<f32>(median_channel(center,0u),median_channel(center,1u),median_channel(center,2u),local(center).a);
    }
    var sum=vec3<f32>(0.0); var weights=0.0;
    var extreme=select(vec3<f32>(3.402823e38),vec3<f32>(-3.402823e38),p.variant==5u);
    for(var y=-16;y<=16;y++){if abs(y)>radius{continue;} for(var x=-16;x<=16;x++){if abs(x)>radius{continue;}
        let value=local(center+vec2<i32>(x,y)).rgb;
        if p.variant==5u{extreme=max(extreme,value);}else if p.variant==6u{extreme=min(extreme,value);}else{
            var weight=1.0;if p.variant==0u{let sigma=max(f32(radius)*0.5,0.5);weight=exp(-f32(x*x+y*y)/(2.0*sigma*sigma));}
            sum+=value*weight;weights+=weight;
        }
    }}
    let rgb=select(sum/max(weights,0.000001),extreme,p.variant==5u||p.variant==6u);
    return vec4<f32>(rgb,local(center).a);
}
fn srgb_linear(v:vec3<f32>)->vec3<f32>{return select(pow((v+vec3<f32>(0.055))/1.055,vec3<f32>(2.4)),v/12.92,v<=vec3<f32>(0.04045));}
fn linear_srgb(v:vec3<f32>)->vec3<f32>{return select(1.055*pow(v,vec3<f32>(1.0/2.4))-vec3<f32>(0.055),v*12.92,v<=vec3<f32>(0.0031308));}
fn algebra(values:array<f32,3>)->f32 {
    var stack:array<f32,32>; var top=0u;
    for(var i=0u;i<32u;i++){
        if i>=p.point_count{break;}
        let instruction=p.points[i]; let op=u32(round(instruction.x));
        if op<=2u{stack[top]=values[op];top+=1u;}
        else if op==3u{stack[top]=instruction.y;top+=1u;}
        else if op>=9u {
            if top==0u{return 0.0;} let a=stack[top-1u];
            if op==9u{stack[top-1u]=-a;}else if op==10u{stack[top-1u]=sin(a);}
            else if op==11u{stack[top-1u]=cos(a);}else if op==12u{stack[top-1u]=abs(a);}
            else{stack[top-1u]=sqrt(max(a,0.0));}
        } else {
            if top<2u{return 0.0;}let b=stack[top-1u];let a=stack[top-2u];top-=1u;
            if op==4u{stack[top-1u]=a+b;}else if op==5u{stack[top-1u]=a-b;}
            else if op==6u{stack[top-1u]=a*b;}else if op==7u{stack[top-1u]=select(a/b,0.0,abs(b)<=0.000001);}
            else{stack[top-1u]=pow(max(a,0.0),b);}
        }
    }
    if top!=1u{return 0.0;}return stack[0];
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let size=textureDimensions(output_tex);if id.x>=size.x||id.y>=size.y{return;}
    let q=id.xy;let a=load_a(q,size);var r=a;
    switch p.operation {
        case 1u:{r=vec4<f32>(curve(a.r),curve(a.g),curve(a.b),a.a);}
        case 2u:{r=vec4<f32>(math(a.r,p.values.x,p.variant),math(a.g,p.values.x,p.variant),math(a.b,p.values.x,p.variant),a.a);}
        case 3u:{r=vec4<f32>(select(vec3<f32>(0.0),vec3<f32>(1.0),a.rgb>=vec3<f32>(p.values.x)),a.a);}
        case 4u:{let half=p.values.y*0.5;r=vec4<f32>(smoothstep(vec3<f32>(p.values.x-half),vec3<f32>(p.values.x+half),a.rgb),a.a);}
        case 5u:{r=apply_filter(q,size);}
        case 6u:{let b=load_b(q,size);if p.variant==8u{var alpha=p.values.x;if(p.connected&4u)!=0u{alpha=load_c(q,size).r;}r=clamp(alpha,0.0,1.0)*a+(1.0-clamp(alpha,0.0,1.0))*b;}else{r=vec4<f32>(math(a.r,b.r,p.variant),math(a.g,b.g,p.variant),math(a.b,b.b,p.variant),math(a.a,b.a,p.variant));}}
        case 7u:{let source_space=p.variant>>16u;let target_space=p.variant&65535u;if source_space!=target_space{r=vec4<f32>(select(srgb_linear(a.rgb),linear_srgb(a.rgb),source_space==1u),a.a);}}
        case 8u:{let v=a[min(p.variant,3u)];r=vec4<f32>(v,v,v,1.0);}
        case 9u:{var gray=dot(a.rgb,vec3<f32>(0.2126,0.7152,0.0722));if p.variant==1u{gray=(a.r+a.g+a.b)/3.0;}if p.variant==2u{gray=(max(a.r,max(a.g,a.b))+min(a.r,min(a.g,a.b)))*0.5;}r=vec4<f32>(gray,gray,gray,a.a);}
        case 10u:{let b=load_b(q,size);let c=load_c(q,size);let d=load_d(q,size);r=vec4<f32>(select(0.0,a.r,(p.connected&1u)!=0u),select(0.0,b.r,(p.connected&2u)!=0u),select(0.0,c.r,(p.connected&4u)!=0u),select(1.0,d.r,(p.connected&8u)!=0u));}
        case 11u:{r=clamp(a,vec4<f32>(0.0),vec4<f32>(1.0));}
        case 12u:{let b=load_b(q,size);let c=load_c(q,size);r=vec4<f32>(
            algebra(array<f32,3>(a.r,b.r,c.r)),algebra(array<f32,3>(a.g,b.g,c.g)),
            algebra(array<f32,3>(a.b,b.b,c.b)),algebra(array<f32,3>(a.a,b.a,c.a)));}
        default:{}
    }
    textureStore(output_tex,vec2<i32>(q),r);
}
