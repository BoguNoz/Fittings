import {compositeStore} from "./stores/composite-store.ts";
import {appComposites} from "./repositories/composites.ts";
import Form from "./components/layout/FormSidePanel.tsx";
import {formStore} from "./stores/form-store.ts";
import Dashboard from "./components/layout/Dashboard.tsx";
import {Toaster} from "@bogunoz/simplify";

const App = () => {

    compositeStore.initializeComposite(appComposites)

  return (
      <>
          <Form
              compositeStore={compositeStore}
              formStore={formStore}
          />
          <div className="flex min-h-screen items-center justify-center">
              <Dashboard
                  compositeStore={compositeStore}
                  formStore={formStore}
              />
          </div>
          <Toaster richColors position="top-right" />
      </>


  )
}

export default App
