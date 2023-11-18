#install.packages("pROC")
library(pROC) # 加载pROC包
rt=read.csv("test total ROC.csv",header=T,sep=",") 

roc1 <- roc(rt$X1year, rt$MobileV3PH,
            ci=TRUE,levels=c("0", "1")) 
roc2 <- roc(rt$X1year, rt$MobileV3,
            ci=TRUE,levels=c("0", "1")) 
roc3 <- roc(rt$X1year, rt$MobileV2PH,
            ci=TRUE,levels=c("0", "1")) 
roc4 <- roc(rt$X1year, rt$MobileV2,
            ci=TRUE,levels=c("0", "1")) 
roc5 <- roc(rt$X1year, rt$AFS,
            ci=TRUE,levels=c("1", "0")) 
roc6 <- roc(rt$X1year, rt$CSGE,
            ci=TRUE,levels=c("1", "0")) 
roc7 <- roc(rt$X1year, rt$Endometrial.thickness,
            ci=TRUE,levels=c("0", "1")) 

plot(roc1,col="#0C00FA") # 绘制ROC曲线
plot(roc2,col="#04D7D9",add = TRUE) # 添加ROC曲线到现有图形上
plot(roc3,col="#2FF011", add = TRUE)
plot(roc4,col="#D9B904", add = TRUE)
plot(roc5,col="#FC6705", add = TRUE)
plot(roc6,col="#BC01FA", add = TRUE)
plot(roc7,col="#FC0B05", add = TRUE)

legend(locator(n=1),legend=c("MobileV3PH","MobileV3","MobileV2PH","MobileV2","AFS","CSGE","Endometrial Thickness"),
       lty=1,col=c("#0C00FA","#04D7D9","#2FF011","#D9B904","#FC6705","#BC01FA","#FC0B05"))

P1 <- plot.roc(rt$gold, rt$pred,
               ci=TRUE, print.auc=TRUE) 
roc.test(roc6,roc7)
rocthr <- ci(roc2, of="thresholds", thresholds="best")

